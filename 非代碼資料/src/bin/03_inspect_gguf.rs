//! 第 03 章：解剖一份真實的 GGUF——你那份 rwkv7-g1k-2.9b-Q8_0。
//!
//! 目的：把「GGUF 是黑箱」變成「GGUF 就是 header + 對齊 + 一堆位元組」。
//!
//! 執行：cargo run --release --bin 03_inspect_gguf
//!       cargo run --release --bin 03_inspect_gguf -- 自訂路徑.gguf

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_quant_tutorial::{gguf_29b, human_bytes};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<()> {
    // 有給路徑就用給的，否則用 2.9B 那份。
    let path = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(gguf_29b);
    // 只讀，不需要 GPU。
    let dev = Device::Cpu;

    println!("檔案：{}", path.display());
    // 檔案實際大小，等一下要跟「宣稱的大小」對帳。
    let file_size = std::fs::metadata(&path)?.len() as usize;
    println!("大小：{}\n", human_bytes(file_size));

    // 開檔。BufReader 是因為 header 是大量小讀取。
    let mut file = BufReader::new(File::open(&path)?);
    // 解析 header：magic、版本、中繼資料、張量資訊表，以及資料區起點。
    let content = gguf_file::Content::read(&mut file)?;

    // ── 中繼資料 ────────────────────────────────────────────────────────
    println!("【中繼資料】共 {} 筆", content.metadata.len());
    // BTreeMap 排序，讓輸出穩定可讀。
    let sorted: BTreeMap<&String, &gguf_file::Value> = content.metadata.iter().collect();
    for (k, v) in sorted {
        // 詞表動輒六萬多個字串，印出來會洗版，只印長度。
        let shown = match v {
            gguf_file::Value::Array(a) if a.len() > 16 => format!("[陣列，{} 項]", a.len()),
            gguf_file::Value::String(s) if s.len() > 60 => {
                format!("{:?}…", &s[..60.min(s.len())])
            }
            other => format!("{other:?}"),
        };
        println!("  {k} = {shown}");
    }

    // ── 張量清單 ────────────────────────────────────────────────────────
    println!("\n【張量】共 {} 個", content.tensor_infos.len());

    // 依型別統計：(張量數, 位元組數)。
    let mut by_dtype: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    // 依「張量名前綴」統計，可以看出權重分佈在哪裡。
    let mut by_group: BTreeMap<String, usize> = BTreeMap::new();
    // 總位元組數。
    let mut total = 0usize;

    for (name, info) in &content.tensor_infos {
        // 這個張量佔幾個位元組 = 元素數 ÷ 區塊大小 × 每區塊位元組。
        let elems = info.shape.elem_count();
        let bytes = elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        total += bytes;

        // 型別統計。
        let e = by_dtype.entry(format!("{:?}", info.ggml_dtype)).or_insert((0, 0));
        e.0 += 1;
        e.1 += bytes;

        // 群組統計：取名字第一段。llama.cpp 的命名慣例是 `blk.0.att...`、
        // `token_embd.weight`、`output_norm.weight`、`head.weight`。
        // 把 32 層的 `blk.N` 全部歸成一類，否則會印出 32 列幾乎一樣的數字。
        let group = match name.split('.').next().unwrap_or("?") {
            // 所有層級權重合併成一列。
            "blk" => "blk.*（32 層合計）".to_string(),
            // 其餘原樣保留。
            other => other.to_string(),
        };
        *by_group.entry(group).or_insert(0) += bytes;
    }

    println!("\n  型別分佈：");
    for (dtype, (count, bytes)) in &by_dtype {
        println!(
            "    {dtype:<6} {count:>5} 個  {:>12}  ({:.1}%)",
            human_bytes(*bytes),
            *bytes as f64 * 100.0 / total as f64
        );
    }

    println!("\n  區塊分佈：");
    for (group, bytes) in &by_group {
        println!(
            "    {group:<12} {:>12}  ({:.1}%)",
            human_bytes(*bytes),
            *bytes as f64 * 100.0 / total as f64
        );
    }

    // 對帳：張量資料總和 + header 應該約等於檔案大小。
    println!("\n  張量資料合計 {:>12}", human_bytes(total));
    println!("  檔案實際大小 {:>12}", human_bytes(file_size));
    println!(
        "  header + 對齊填充 {:>10}",
        human_bytes(file_size.saturating_sub(total))
    );

    // ── 實際載入一個張量，證明「讀得出來」 ──────────────────────────────
    println!("\n【實測載入】挑一個權重，反量化後看統計量");
    // 挑第 0 層的 value 投影：這是 RWKV-7 裡最「重」的矩陣之一。
    // 因為兩套命名方言並存，這裡逐一嘗試，取第一個存在的名字。
    // 這個小技巧在實務上很常用：同一支工具要能吃 llama.cpp 與 candle 兩種 GGUF。
    let candidates = [
        // llama.cpp 方言。
        "blk.0.time_mix_value.weight",
        // candle safetensors 方言（第 02 章產出的檔案就是這種）。
        "blocks.0.att.value.weight",
        // 更舊的 ggml 方言。
        "blocks.0.att.value",
    ];
    // 找出第一個真的存在的名字。
    let probe = candidates
        .iter()
        .find(|n| content.tensor_infos.contains_key(**n))
        .copied()
        // 三個都沒有就報錯，訊息要能告訴使用者去哪裡找正確名字。
        .ok_or_else(|| anyhow::anyhow!("這份檔案裡找不到第 0 層的 value 權重"))?;
    // 從檔案讀成 QTensor（此時還是壓縮狀態，記憶體佔用只有 1/4）。
    let qt = content.tensor(&mut file, probe, &dev)?;
    println!(
        "  {probe}: 形狀 {:?}，型別 {:?}，壓縮後 {}",
        qt.shape().dims(),
        qt.dtype(),
        human_bytes(qt.storage_size_in_bytes())
    );

    // 反量化成 f32，才能算統計量。
    let t = qt.dequantize(&dev)?;
    // 平均值：量化良好的權重應該接近 0。
    let mean = t.mean_all()?.to_scalar::<f32>()?;
    // 標準差：用來檢查有沒有「整層塌掉」。
    let std = (t.sqr()?.mean_all()?.to_scalar::<f32>()? - mean * mean).sqrt();
    // 最小值 / 最大值：看動態範圍。
    let min = t.min_all()?.to_scalar::<f32>()?;
    let max = t.max_all()?.to_scalar::<f32>()?;
    println!("  平均 {mean:+.5}，標準差 {std:.5}，範圍 [{min:+.4}, {max:+.4}]");

    // 印出前 8 個數值，讓人對「解量化後的實際數字」有感覺。
    let head: Vec<f32> = t.flatten_all()?.narrow(0, 0, 8)?.to_vec1()?;
    println!("  前 8 個值：{head:?}");

    // ── 結論段 ──────────────────────────────────────────────────────────
    println!("\n【重點】");
    println!("  1. GGUF 沒有「壓縮」，只有「換一種表示法」：每個數值仍是固定位元數，");
    println!("     所以可以直接 mmap、可以隨機存取某一個張量，不必整檔解壓。");
    println!("  2. 32 位元組對齊是硬性要求：GPU 與 SIMD 都要靠它做向量化載入。");
    println!("  3. 詞嵌入 + 輸出頭通常佔掉一半以上的體積，這就是為什麼");
    println!("     llama.cpp 預設把它們留在較高精度（Q6_K），而不是跟主體一起壓到 Q4。");

    // ── 名稱方言：為什麼 candle 讀不了這份檔案 ──────────────────────────
    println!("\n【名稱方言】同一份權重，兩套命名");
    println!("  這份 GGUF 用的是 llama.cpp 的張量名；你的 safetensors 用的是另一套。");
    println!("  對照表（左：candle safetensors，右：llama.cpp GGUF）：");
    // 每一列都是實際比對出來的，不是查文件抄的。
    let mapping = [
        ("emb.weight", "token_embd.weight"),
        ("head.weight", "output.weight"),
        ("blocks.N.ln0", "token_embd_norm"),
        ("blocks.N.ln1", "blk.N.attn_norm"),
        ("blocks.N.ln2", "blk.N.attn_norm_2"),
        ("blocks.N.att.receptance.weight", "blk.N.time_mix_receptance.weight"),
        ("blocks.N.att.key.weight", "blk.N.time_mix_key.weight"),
        ("blocks.N.att.value.weight", "blk.N.time_mix_value.weight"),
        ("blocks.N.att.output.weight", "blk.N.time_mix_output.weight"),
        ("blocks.N.att.ln_x", "blk.N.time_mix_ln"),
        ("blocks.N.att.x_r/x_w/x_k/x_v/x_a/x_g", "blk.N.time_mix_lerp_fused（六個融成一個）"),
        ("blocks.N.att.w0/w1/w2", "blk.N.time_mix_w0/w1/w2"),
        ("blocks.N.att.a0/a1/a2", "blk.N.time_mix_a0/a1/a2"),
        ("blocks.N.att.v0/v1/v2", "blk.N.time_mix_v0/v1/v2"),
        ("blocks.N.ffn.key.weight", "blk.N.channel_mix_key.weight"),
        ("blocks.N.ffn.value.weight", "blk.N.channel_mix_value.weight"),
        ("blocks.N.ffn.x_k", "blk.N.channel_mix_lerp_k"),
        ("ln_out", "output_norm"),
    ];
    for (candle_name, llama_name) in mapping {
        println!("    {candle_name:<34} → {llama_name}");
    }
    println!("\n  兩個後果：");
    println!("    1. 你那份 2.9B Q8_0 丟給 candle 的 VarBuilder::from_gguf，");
    println!("       會因為找不到 `blocks.0.att.receptance.weight` 而失敗——");
    println!("       必須先寫一層名稱映射（第 03 章給了 20 行的實作）。");
    println!("    2. 反過來，用第 02 章產出的 GGUF 給 llama.cpp 也讀不動，");
    println!("       除了名字不對，還缺 tokenizer.ggml.tokens 那 65536 個字串。");

    // 提醒 candle 的缺口——這是第三章的主題。
    println!("\n【注意】candle 0.11 沒有 quantized_rwkv_v7 模組。");
    println!("  這份 Q8_0 你可以讀、可以逐張量反量化，但沒有現成的 forward 可以跑。");
    println!("  要用 Ollama（llama.cpp 後端）才跑得動 2.9B。詳見文件三。");

    // 讓編譯器不要抱怨 Tensor 未使用（上面用到了，但保留這行當說明）。
    let _ = Tensor::zeros(1, candle_core::DType::F32, &dev)?;

    Ok(())
}
