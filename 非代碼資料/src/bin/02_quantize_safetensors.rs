//! 第 02 章：把 safetensors（BF16）轉成 GGUF（Q8_0 / Q4_K），全程 CPU。
//!
//! 這是「自己動手量化」的完整流程，沒有 llama.cpp、沒有 Python，只有 candle。
//!
//! 執行：
//!   cargo run --release --bin 02_quantize_safetensors -- --dtype q8_0
//!   cargo run --release --bin 02_quantize_safetensors -- --dtype q4k --src 1.5b

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use candle_quant_tutorial::{gguf_29b, human_bytes, mid_15b, small_01b};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

/// 命令列參數。只有三個，不值得為它拉一個 clap 進來。
struct Args {
    /// 來源 safetensors 路徑。
    src: std::path::PathBuf,
    /// 輸出 GGUF 路徑。
    dst: std::path::PathBuf,
    /// 主體張量要用的量化型別。
    dtype: GgmlDType,
    /// 寫進 GGUF 的模型名稱（llama.cpp / Ollama 會顯示這個）。
    name: String,
    /// 層數，寫進 rwkv7.block_count。
    layers: usize,
    /// 隱藏維度，寫進 rwkv7.embedding_length。
    hidden: usize,
    /// 上下文長度，寫進 rwkv7.context_length。
    ctx: u64,
}

/// 手寫的參數解析。`--key value` 的形式，認不得的就報錯，避免打錯字卻靜默生效。
fn parse_args() -> Result<Args> {
    // 先給 0.1B 的預設值——這是本教學的主線模型。
    let mut args = Args {
        src: small_01b(),
        dst: std::path::PathBuf::from("rwkv7-0.1b-q8_0.gguf"),
        dtype: GgmlDType::Q8_0,
        name: "rwkv7-g1d-0.1b".to_string(),
        layers: 12,
        hidden: 768,
        ctx: 8192,
    };

    // 收集 argv，跳過第 0 個（程式自己的路徑）。
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // 用索引走，因為每個旗標都要吃掉後面那個值。
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            // 切到 1.5B：形狀與檔名一起換，避免只換一半造成維度不符。
            "--src" if argv.get(i + 1).map(|s| s.as_str()) == Some("1.5b") => {
                args.src = mid_15b();
                args.name = "rwkv7-g1d-1.5b".to_string();
                args.layers = 24;
                args.hidden = 2048;
                i += 2;
            }
            // 自訂輸出路徑。
            "--out" => {
                args.dst = argv.get(i + 1).context("--out 需要一個路徑")?.into();
                i += 2;
            }
            // 選量化型別。只開放這三種：Q8_0 求穩、Q4_K 求小、F16 當對照組。
            "--dtype" => {
                args.dtype = match argv.get(i + 1).map(|s| s.as_str()) {
                    Some("q8_0") => GgmlDType::Q8_0,
                    Some("q4k") => GgmlDType::Q4K,
                    Some("q6k") => GgmlDType::Q6K,
                    Some("f16") => GgmlDType::F16,
                    other => bail!("--dtype 只接受 q8_0 / q4k / q6k / f16，收到 {other:?}"),
                };
                i += 2;
            }
            // 其他一律視為錯誤，不猜。
            other => bail!("未知參數 {other}"),
        }
    }
    Ok(args)
}

/// 決定「這一個張量該用哪種量化型別」。
///
/// 這是整個流程最需要判斷力的地方，也是 llama.cpp 的 `--tensor-type` 在做的事。
fn pick_dtype(name: &str, shape: &[usize], default: GgmlDType) -> GgmlDType {
    // 只看二維張量。一維（LayerNorm 的 weight/bias）與三維（RWKV 的 x_r/x_w 之類）
    // 元素太少，量化省不下空間，反而增加解量化開銷，業界慣例是保留 F16/F32。
    if shape.len() != 2 {
        return GgmlDType::F16;
    }
    // 最後一維必須能被區塊大小整除，否則量化函式會直接報錯。
    let last = *shape.last().unwrap();
    // 詞嵌入與輸出頭特別敏感：embedding 是查表（誤差不會被平均掉），
    // head 直接決定 logits。這兩者一律給更高精度，這也是 Qwen / DeepSeek 的通行做法。
    let is_sensitive = name.contains("emb.weight") || name.contains("head.weight");
    if is_sensitive {
        // Q6_K 需要 256 的倍數；65536 與 768 都滿足，但仍寫成判斷式以防其他模型。
        if last % 256 == 0 {
            return GgmlDType::Q6K;
        }
        // 退一步用 Q8_0（只需要 32 的倍數）。
        return GgmlDType::Q8_0;
    }
    // 一般張量：檢查能不能被目標型別的區塊大小整除。
    if last % default.block_size() != 0 {
        // 不能整除就退回 F16。Q4_K 的 256 很嚴格，所以 w1/g1 這類小矩陣
        // （最後一維 64 / 128）在 Q4_K 下必然落到這個分支。
        return GgmlDType::F16;
    }
    // 一切合格，用使用者指定的型別。
    default
}

fn main() -> Result<()> {
    // 解析參數。
    let args = parse_args()?;
    // 量化在 CPU 做：CUDA 版量化 kernel 存在，但 CPU 版本行為最可預測，
    // 而且 safetensors 是 mmap 進來的，本來就在主記憶體。
    let dev = Device::Cpu;

    println!("來源：{}", args.src.display());
    println!("輸出：{}", args.dst.display());
    println!("主體型別：{:?}\n", args.dtype);

    // 計時起點。
    let t0 = Instant::now();

    // 一次把整個 safetensors 讀進記憶體。0.1B 是 382 MB，1.5B 是 3 GB，
    // 你的 16 GB 記憶體都吃得下；再大的模型就得改用 mmap 分批處理了。
    let tensors: std::collections::HashMap<String, Tensor> =
        candle_core::safetensors::load(&args.src, &dev)
            .with_context(|| format!("讀不到 {}", args.src.display()))?;

    println!("讀入 {} 個張量，耗時 {:.1}s", tensors.len(), t0.elapsed().as_secs_f32());

    // 依名字排序，讓輸出可重現（HashMap 的迭代順序是隨機的）。
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    // 量化後的張量先收集起來，最後一次性寫檔。
    let mut quantized: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    // 統計：型別 → (張量數, 位元組數)，用 BTreeMap 讓輸出順序穩定。
    let mut stats: BTreeMap<String, (usize, usize)> = BTreeMap::new();

    for name in names {
        // 取出原始張量。
        let t = &tensors[name];
        // 原始形狀，用來判斷 rank 與最後一維。
        let shape: Vec<usize> = t.dims().to_vec();
        // 決定這個張量要用哪種型別。
        let dtype = pick_dtype(name, &shape, args.dtype);

        // 量化。QTensor::quantize 內部會先轉 f32 再 flatten，
        // 所以 BF16 → 量化的過程不會有額外的精度損失。
        let qt = QTensor::quantize(t, dtype)
            .with_context(|| format!("量化 {name}（{shape:?} → {dtype:?}）失敗"))?;

        // 累計統計。
        let entry = stats.entry(format!("{dtype:?}")).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += qt.storage_size_in_bytes();

        // 存起來。
        quantized.push((name.clone(), qt));
    }

    // ── 組 GGUF 中繼資料 ────────────────────────────────────────────────
    // 這些 key 不是我發明的，是 llama.cpp 的 rwkv7 載入器認的欄位。
    // 我是從你那份 rwkv7-g1k-2.9b-Q8_0.gguf 的 header 一個一個抄下來對照的。
    let meta: Vec<(String, Value)> = vec![
        // 架構名稱，llama.cpp 靠它選載入器。
        ("general.architecture".into(), Value::String("rwkv7".into())),
        // 檔案類型標記，固定 model。
        ("general.type".into(), Value::String("model".into())),
        // 顯示名稱。
        ("general.name".into(), Value::String(args.name.clone())),
        // 尺寸標籤，Ollama 的 `ollama list` 會顯示這個。
        ("general.size_label".into(), Value::String("0.1B".into())),
        // 上下文長度。
        ("rwkv7.context_length".into(), Value::U64(args.ctx)),
        // 隱藏維度。
        ("rwkv7.embedding_length".into(), Value::U64(args.hidden as u64)),
        // 層數。
        ("rwkv7.block_count".into(), Value::U64(args.layers as u64)),
        // LayerNorm 的 eps。candle 的 rwkv_v7 寫死 1e-5，這裡必須一致。
        ("rwkv7.attention.layer_norm_epsilon".into(), Value::F32(1e-5)),
        // 每個注意力頭的維度。G1 全系列都是 64。
        ("rwkv7.wkv.head_size".into(), Value::U64(64)),
        // 以下四個是 RWKV-7 特有的低秩分解維度，載入器要靠它們推形狀。
        ("rwkv7.attention.decay_lora_rank".into(), Value::U64(64)),
        ("rwkv7.attention.iclr_lora_rank".into(), Value::U64(64)),
        ("rwkv7.attention.value_residual_mix_lora_rank".into(), Value::U64(32)),
        ("rwkv7.attention.gate_lora_rank".into(), Value::U64(128)),
        // FFN 中間維度 = hidden × 4。
        ("rwkv7.feed_forward_length".into(), Value::U64((args.hidden * 4) as u64)),
        // 量化格式版本，2 是現行版本。
        ("general.quantization_version".into(), Value::U32(2)),
        // 檔案類型編號：7 = Q8_0、12 = Q4_K、14 = Q6_K、1 = F16（GGUF 規格定義）。
        (
            "general.file_type".into(),
            Value::U32(match args.dtype {
                GgmlDType::Q8_0 => 7,
                GgmlDType::Q4K => 12,
                GgmlDType::Q6K => 14,
                _ => 1,
            }),
        ),
    ];

    // ── 寫檔 ────────────────────────────────────────────────────────────
    // 建立輸出檔。用 BufWriter 是因為 GGUF 是一堆小片段連續寫，沒有緩衝會很慢。
    let mut file = BufWriter::new(File::create(&args.dst)?);

    // candle 的 write 要求 &[(&str, &Value)] 與 &[(&str, &QTensor)]，
    // 所以先做兩層借用視圖。
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = quantized
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    // 真正寫入。內部會自動做 32 位元組對齊（GGUF 規格要求）。
    gguf_file::write(&mut file, &meta_refs, &tensor_refs)?;

    // 釋放。
    drop(file);

    // ── 結算 ────────────────────────────────────────────────────────────
    println!("\n型別分佈：");
    // 原始 safetensors 大小。
    let src_size = std::fs::metadata(&args.src)?.len() as usize;
    // 輸出 GGUF 大小。
    let dst_size = std::fs::metadata(&args.dst)?.len() as usize;
    for (dtype, (count, bytes)) in &stats {
        println!("  {dtype:<6} {count:>4} 個張量  {:>12}", human_bytes(*bytes));
    }
    println!("\n原始 {:>12}", human_bytes(src_size));
    println!("輸出 {:>12}", human_bytes(dst_size));
    println!("壓縮比 {:.2}x", src_size as f64 / dst_size as f64);
    println!("總耗時 {:.1}s", t0.elapsed().as_secs_f32());

    // 提醒一件很重要的事：這份 GGUF 給 candle 讀完全沒問題，
    // 但直接丟給 llama.cpp 可能會有相容性問題，原因見文件第三章「三個坑」。
    println!(
        "\n提示：這份檔案給 candle 讀沒問題；要給 llama.cpp / Ollama 用，\n\
         還需要補 tokenizer.ggml.tokens 等詞表欄位（見第三章）。"
    );

    // 順手印出對照路徑，方便下一步。
    println!("對照組（別人量化好的）：{}", gguf_29b().display());

    Ok(())
}
