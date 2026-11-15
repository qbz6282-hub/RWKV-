//! 第 04 章：在 CUDA 上微調 rwkv7-g1d-0.1b（全量參數）。
//!
//! 這是「能真的跑起來」的最小完整訓練迴圈：載入權重 → 建圖 → 前向 → 交叉熵
//! → 反傳 → 梯度裁剪 → AdamW 更新 → 存檔。
//!
//! 執行：
//!   cargo run --release --features cuda --bin 04_train_full
//!   cargo run --release --features cuda --bin 04_train_full -- --steps 50 --chunk 64
//!
//! 沒有 CUDA 也能編譯與執行，只是會退回 CPU（慢 20~50 倍，用來驗證邏輯）。

use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{loss, Init, VarBuilder, VarMap};
use candle_transformers::models::rwkv_v7::{Config, Model, ModelVersion, State};
use candle_quant_tutorial::small_01b;
use std::time::Instant;

/// 一個「會轉接 get_unchecked」的 VarMap 後端。
///
/// 為什麼需要它？這是本章最關鍵的坑：
/// candle 的 `rwkv_v7::Model::new` 為了自動推導 LoRA 維度，對 `w1/a1/v1/g1`
/// 呼叫了 `vb.get_unchecked(name)`——它只給名字、不給形狀。
/// 而 `VarMap` 這個後端沒有實作 `get_unchecked`，直接報錯：
/// 「`get_unchecked` does not make sense for `VarMap`, use `get`」。
///
/// 也就是說：**用 `VarBuilder::from_varmap` 根本建不出 rwkv_v7 模型**。
///
/// 解法有兩條：
///   A. fork `candle-transformers/src/models/rwkv_v7.rs`，把 4 處 `get_unchecked` 改成 `get`。
///   B. 就是這裡——`SimpleBackend` 是公開 trait，`VarBuilder::from_backend` 也是公開 API，
///      所以我們自己包一層：`get` 原樣轉給 VarMap，`get_unchecked` 用「查表取名」補上。
///
/// B 的好處是零 fork、跟著上游升級不會痛。這也是 candle 唯一留給使用者的縫。
struct VarmapBackend {
    /// 真正的參數倉庫。
    map: VarMap,
}

impl SimpleBackend for VarmapBackend {
    /// 有給形狀的一般取用：直接轉給 VarMap，它會檢查形狀是否相符。
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        self.map.get(s, name, h, dtype, dev)
    }

    /// 只給名字的取用：VarMap 沒實作，我們自己查。
    /// 因為權重是從 safetensors 原樣載入的，形狀一定正確，不需要再檢查。
    fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> CandleResult<Tensor> {
        // 取鎖查表。
        let data = self.map.data().lock().unwrap();
        // 找不到就明確報錯，訊息帶上名字，比 upstream 的通用錯誤好查。
        match data.get(name) {
            // as_tensor() 回傳的是 Var 內部的張量視圖，帶梯度追蹤。
            Some(var) => Ok(var.as_tensor().clone()),
            None => candle_core::bail!("VarmapBackend: 找不到參數 {name}"),
        }
    }

    /// 模型載入時會用這個判斷「某個可選張量存不存在」（例如 v7a 的 s_emb_x）。
    fn contains_tensor(&self, name: &str) -> bool {
        self.map.data().lock().unwrap().contains_key(name)
    }
}

/// 訓練超參數。全部集中在一處，方便你改。
struct TrainArgs {
    /// 梯度累積的序列長度，也就是 TBPTT 的「截斷長度」。
    chunk: usize,
    /// 總共跑幾個 optimizer step。
    steps: usize,
    /// 學習率。全量微調用 1e-5 這種量級，比從零訓練小兩個數量級。
    lr: f64,
    /// 梯度全域範數上限。RWKV 的衰減項偶爾會爆，這個一定要有。
    clip: f32,
    /// 存檔路徑。
    out: std::path::PathBuf,
}

fn parse_args() -> TrainArgs {
    // 預設值針對 6 GB 顯存調過：chunk 32 的活化記憶體很小，
    // 主要開銷在參數 + 梯度 + AdamW 的兩個動量緩衝（共 4 份權重）。
    let mut a = TrainArgs {
        chunk: 32,
        steps: 20,
        lr: 1e-4,
        clip: 10.0,
        out: std::path::PathBuf::from("rwkv7-0.1b-ft.safetensors"),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        // 每個旗標都吃掉下一個值，所以 i 每次加 2。
        match argv[i].as_str() {
            "--chunk" => {
                a.chunk = argv[i + 1].parse().unwrap_or(32);
                i += 2;
            }
            "--steps" => {
                a.steps = argv[i + 1].parse().unwrap_or(20);
                i += 2;
            }
            "--lr" => {
                a.lr = argv[i + 1].parse().unwrap_or(1e-5);
                i += 2;
            }
            "--clip" => {
                a.clip = argv[i + 1].parse().unwrap_or(10.0);
                i += 2;
            }
            "--out" => {
                a.out = argv[i + 1].clone().into();
                i += 2;
            }
            _ => i += 1,
        }
    }
    a
}

/// 計算梯度全域 L2 範數，必要時就地縮放。
///
/// candle 沒有內建 clip_grad_norm，得自己寫——這其實是好事，
/// 寫一次就懂 PyTorch 那個函式到底在做什麼。
fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f32) -> Result<f32> {
    // 先把所有梯度平方和累加起來。
    let mut sum_sq = 0f32;
    for v in vars {
        // GradStore 用張量 id 查梯度；沒有梯度代表這個參數沒參與計算（例如被凍結）。
        if let Some(g) = grads.get(v.as_tensor()) {
            // 平方後全部加總，轉成純量。
            sum_sq += g.sqr()?.sum_all()?.to_scalar::<f32>()?;
        }
    }
    // 開根號得到 L2 範數。
    let norm = sum_sq.sqrt();
    // 超過上限才縮放，沒超過就原樣放行（這是 clip 而不是 normalize）。
    if norm > max_norm {
        // 縮放係數。加 1e-6 是為了避免除以零。
        let scale = (max_norm / (norm + 1e-6)) as f64;
        for v in vars {
            // 逐一取出、乘上係數、再塞回去。
            if let Some(g) = grads.get(v.as_tensor()) {
                // 這裡必須用 GradStore::insert 覆寫，因為 Optimizer::step 讀的是 store 裡的值。
                grads.insert(v.as_tensor(), (g * scale)?);
            }
        }
    }
    Ok(norm)
}

/// 把一段文字轉成 RWKV World 詞表的 token id。
///
/// RWKV World 的前 256 個詞元就是原始位元組（token 1 = 0x00, token 2 = 0x01 …），
/// 所以 `byte + 1` 就是合法的 token id。這是沒有 encode 函式時的標準替代方案，
/// 也讓我省掉自己實作那個貪婪分詞器的麻煩。
fn encode_bytes(text: &str) -> Vec<u32> {
    text.as_bytes().iter().map(|b| *b as u32 + 1).collect()
}

fn main() -> Result<()> {
    // 參數。
    let args = parse_args();

    // 裝置選擇：有開 cuda feature 就走 GPU，否則退回 CPU。
    // `cuda_if_available` 在沒編 CUDA 時會回傳 Cpu，所以這行兩種情況都安全。
    let dev = if cfg!(feature = "cuda") {
        Device::cuda_if_available(0).context("CUDA 初始化失敗，檢查 nvcc 與驅動版本")?
    } else {
        Device::Cpu
    };
    println!("裝置：{dev:?}");

    // ── 1. 把權重載進 VarMap ────────────────────────────────────────────
    // VarMap 是 candle 的「參數倉庫」：它是 HashMap<String, Var>，
    // 而 Var 是可被梯度追蹤、可被優化器更新的張量。
    // 注意不需要 mut：填充參數是透過 data() 拿到內部 Mutex，不是改 VarMap 本身。
    let varmap = VarMap::new();

    // 先從 safetensors 讀出所有張量（BF16）。
    let raw = candle_core::safetensors::load(small_01b(), &dev)?;
    println!("讀入 {} 個張量（BF16）", raw.len());

    // 轉成 F32 再放進 VarMap。這一步是刻意的：
    // AdamW 的動量緩衝若用 BF16 會嚴重掉精度，業界慣例是「BF16 前向 + F32 主權重」。
    // 在 6 GB 顯存上，0.1B 用 F32 主權重完全放得下（764 MB）。
    let f32_tensors: Vec<(String, Tensor)> = raw
        .iter()
        // 逐個轉型別。`to_dtype` 會產生新張量，舊的 BF16 版本仍在 raw 裡，
        // 所以下一步要記得把 raw 釋放掉，否則同時佔兩份記憶體。
        .map(|(k, v)| Ok((k.clone(), v.to_dtype(DType::F32)?)))
        .collect::<Result<Vec<_>>>()?;

    // 放進 VarMap。**這裡有個坑**：
    // `VarMap::set` 只會「更新已存在的變數」，遇到不存在的名字直接 bail，
    // 它不會幫你新增。所以第一次填充必須直接操作底層的 HashMap。
    // （`varmap.data()` 回傳 `&Mutex<HashMap<String, Var>>`，是公開 API。）
    {
        // 取得鎖。作用域寫窄一點，離開這個 block 就釋放，避免後面死鎖。
        let mut data = varmap.data().lock().unwrap();
        for (name, tensor) in &f32_tensors {
            // Var::from_tensor 會複製一份 storage 並標記為「可訓練」，
            // 所以原本的 f32_tensors 可以安全丟掉。
            data.insert(name.clone(), Var::from_tensor(tensor)?);
        }
    }
    // 明確釋放 BF16 原始資料，把記憶體還給系統。
    drop(raw);
    drop(f32_tensors);
    println!("VarMap 內共 {} 個參數張量", varmap.all_vars().len());

    // ── 2. 建模型 ───────────────────────────────────────────────────────
    // 關鍵：VarBuilder 指向我們自己的轉接器，模型裡每個權重都會是 Var 的視圖，
    // 因此 loss.backward() 之後，梯度會自動累積到這些 Var 上。
    //
    // 注意這裡用的是 `from_backend` 而不是 `from_varmap`——原因見上面 VarmapBackend 的說明。
    let vb = VarBuilder::from_backend(
        // 把 varmap 交出去（clone 是 Arc 級別的複製，共用同一份參數）。
        Box::new(VarmapBackend { map: varmap.clone() }),
        // 主權重用 F32。
        DType::F32,
        // 裝置。
        dev.clone(),
    );

    // 0.1B（g1d）的設定，數值全部來自 safetensors 的實際形狀，不是猜的：
    //   emb [65536, 768]、ffn.key [3072, 768]、r_k [12, 64]（→ 12 個頭 × 64）
    let cfg = Config {
        // g1d 是純 v7，沒有 DeepEmbed（v7a）也沒有 DEA（v7b）。
        version: ModelVersion::V7,
        // 詞表大小，RWKV World 固定 65536。
        vocab_size: 65536,
        // 隱藏維度。
        hidden_size: 768,
        // 層數。
        num_hidden_layers: 12,
        // 每頭維度，G1 全系列都是 64。
        head_size: 64,
        // None 代表用 hidden × 4 = 3072，正好符合實際形狀。
        intermediate_size: None,
        // 0 代表關閉「每 N 層縮放」，G1 世代不需要。
        rescale_every: 0,
    };

    // 建模型。注意這裡會做 transpose + contiguous，產生額外約 540 MB 的副本，
    // 這是 candle 為了讓 matmul 走連續記憶體而做的取捨。
    let model = Model::new(&cfg, vb).context("模型建構失敗：形狀與權重不符")?;
    println!("模型建構完成");

    // ── 3. 準備資料 ─────────────────────────────────────────────────────
    // 訓練語料。刻意用繁體中文，因為位元組級分詞對中文是 3 bytes/字，
    // 32 個 token 只涵蓋約 10 個漢字——這正好讓你感受到位元組分詞的代價。
    let corpus = "RWKV 把注意力換成一個固定大小的遞迴狀態，所以推理成本不隨長度成長。\
                  這句話你會在第 32 個詞元之後再看到一次，用來檢查模型有沒有記住。\
                  RWKV 把注意力換成一個固定大小的遞迴狀態，所以推理成本不隨長度成長。";
    // 轉成 token。
    let tokens = encode_bytes(corpus);
    println!("語料 {} 字 → {} 個 token", corpus.chars().count(), tokens.len());

    // ── 4. 優化器 ───────────────────────────────────────────────────────
    // 取出所有 Var。全量微調就是「全部交給優化器」。
    let vars = varmap.all_vars();
    // AdamW 超參數。weight_decay 給 0.01，是 LLM 微調的常見值。
    let params = ParamsAdamW {
        lr: args.lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut opt = AdamW::new(vars.clone(), params)?;
    println!("優化器就緒，管理 {} 個參數", vars.len());

    // ── 5. 訓練迴圈 ─────────────────────────────────────────────────────
    // 這是整份教學的核心：RWKV 是遞迴模型，只能逐 token 前向，
    // 所以「一個 chunk」= 從乾淨狀態開始，連續吃 chunk 個 token。
    // 這就是 TBPTT（Truncated Backpropagation Through Time）：
    // 每個 chunk 之間丟掉狀態，梯度不跨 chunk 累積，記憶體才不會無限成長。
    let t0 = Instant::now();
    // 記錄最近幾步的 loss 用來算移動平均。
    let mut recent: Vec<f32> = Vec::new();

    for step in 0..args.steps {
        // 每個 step 從語料裡取一個 chunk。用 step 當偏移，讓模型看到不同片段。
        let start = (step * args.chunk) % (tokens.len().saturating_sub(args.chunk + 1)).max(1);
        // 輸入：chunk 個 token。
        let input = &tokens[start..start + args.chunk];
        // 目標：往後移一個位置，也就是「預測下一個 token」。
        let target = &tokens[start + 1..start + 1 + args.chunk];

        // 每個 chunk 都重建狀態。這一步等同於 PyTorch 的 `state.detach()`，
        // 但 candle 的 State 沒有 detach 方法，直接重建最乾淨。
        let mut state = State::new(&cfg, &dev)?;

        // 收集每個位置的 logits。
        let mut logits_per_pos: Vec<Tensor> = Vec::with_capacity(args.chunk);
        for (pos, tok) in input.iter().enumerate() {
            // 輸入張量形狀必須是 (batch=1, seq=1)，因為 Model::forward 內部會 squeeze 兩次。
            let xs = Tensor::from_vec(vec![*tok], (1, 1), &dev)?;
            // 前向一個 token。第三個參數是 token id 切片，v7a/v7b 的 DeepEmbed 會用到；
            // 純 v7 傳進去也無害。
            let logits = model.forward(&xs, &mut state, &[*tok])?;
            // 形狀是 (vocab_size,)，收起來。
            logits_per_pos.push(logits);
            // pos 只為了讓編譯器知道我們有在用；實際不需要。
            let _ = pos;
        }

        // 把 chunk 個 (vocab,) 疊成 (chunk, vocab)。
        let logits = Tensor::stack(&logits_per_pos, 0)?;
        // 目標轉成張量，形狀 (chunk,)，dtype 必須是 u32。
        let target_t = Tensor::from_vec(target.to_vec(), args.chunk, &dev)?;

        // 交叉熵：內部會做 log_softmax + nll，數值穩定版。
        let loss = loss::cross_entropy(&logits, &target_t)?;
        // 取出純量值來印。這一步會觸發同步（GPU → CPU），所以放在 backward 之前。
        let loss_v = loss.to_scalar::<f32>()?;

        // 反向傳播。回傳的 GradStore 是「張量 id → 梯度」的映射。
        let mut grads = loss.backward()?;

        // 梯度裁剪。回傳裁剪前的範數，方便觀察是否常態性爆梯度。
        let gnorm = clip_grad_norm(&mut grads, &vars, args.clip)?;

        // 更新參數。
        opt.step(&grads)?;

        // 記錄。
        recent.push(loss_v);
        // 只留最近 5 步。
        if recent.len() > 5 {
            recent.remove(0);
        }
        // 移動平均。
        let avg = recent.iter().sum::<f32>() / recent.len() as f32;

        println!(
            "step {:>3}  loss {:.4}  (avg {:.4})  |g| {:.3}  {:.1}s",
            step + 1,
            loss_v,
            avg,
            gnorm,
            t0.elapsed().as_secs_f32()
        );
    }

    // ── 6. 存檔 ─────────────────────────────────────────────────────────
    // VarMap::save 直接寫 safetensors，key 就是原本的張量名，
    // 所以存出來的檔案可以用 `VarBuilder::from_mmaped_safetensors` 原樣載回。
    varmap.save(&args.out)?;
    println!("\n已存檔：{}", args.out.display());
    println!(
        "大小 {} MB",
        std::fs::metadata(&args.out)?.len() / 1024 / 1024
    );

    // ── 7. 收尾觀察 ─────────────────────────────────────────────────────
    println!("\n【觀察重點】");
    println!("  1. loss 從 11 左右開始（ln(65536) = 11.09 是「完全亂猜」的上限），");
    println!("     前幾步掉得很快，之後趨緩——這是「模型本來就懂中文，只是被你的語料帶偏」。");
    println!("  2. |g| 的絕對值會嚇到你：這個模型在 chunk=8 時是 600~1000 量級。");
    println!("     不是 bug——詞表 65536，softmax 的梯度會在 6.5 萬維上累加，範數自然大。");
    println!("     所以 clip 千萬別設 1.0：那等於把學習率再除以 700，loss 會看起來完全不會動。");
    println!("     本範例預設 10.0，是「該剪的時候剪、不該剪的時候放行」的折衷。");
    println!("  3. chunk 從 32 加到 64，顯存幾乎不變（狀態是固定大小），");
    println!("     但每一步的耗時會線性成長——這正是 RWKV 與 Transformer 的差異所在：");
    println!("     Transformer 的 KV cache 隨長度成長，RWKV 的狀態不變。");
    println!("  4. 全量微調 0.1B 在這張 6 GB 卡上約佔 3.6 GB：");
    println!("     參數 764 MB + 梯度 764 MB + AdamW 兩個動量 1.5 GB + 轉置副本 540 MB。");
    println!("     想微調 1.5B，就得換成 LoRA——見第 05 章。");

    Ok(())
}
