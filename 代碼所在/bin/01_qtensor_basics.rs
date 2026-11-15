//! 第 01 章：QTensor 是什麼、量化誤差長什麼樣。
//!
//! 不讀任何模型檔，3 秒跑完。目的是先建立「一個 QTensor 就是一堆區塊」的直覺，
//! 後面看到 GGUF 才不會覺得是一團黑箱。
//!
//! 執行：cargo run --release --bin 01_qtensor_basics

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_quant_tutorial::{bits_per_weight, dtype_table, human_bytes};

/// 一個極簡的確定性常態隨機數產生器。
///
/// 為什麼不用 `Tensor::randn`？因為 candle 的 `Device::set_seed` 對 CPU 後端
/// 會直接報錯：`cannot seed the CPU rng with set_seed`。
/// 也就是說 CPU 上根本沒辦法固定種子，數字每次都不一樣，文件就沒辦法引用實測值。
/// 自己造一個 xorshift32 + Box-Muller 只要二十行，而且沒有額外依賴。
struct Rng(u32);

impl Rng {
    /// 建立。種子不能為 0（xorshift 的 0 是吸收態，會永遠輸出 0），所以強制設最低位。
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    /// xorshift32：三個移位與 XOR，週期 2³²−1，對我們這種用途綽綽有餘。
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// 取 (0, 1) 的均勻分佈。右移 8 位是因為 f32 的尾數只有 24 bit。
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Box-Muller 變換：兩個獨立的均勻分佈 → 一個標準常態分佈。
    /// max(1e-7) 是為了避免 u1 = 0 時 ln(0) = −∞。
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn main() -> Result<()> {
    // 量化全部在 CPU 上做：GPU 的量化 kernel 只是把同一套數學搬過去，
    // 先用 CPU 看清楚數字，才不會被「跑得比較快」掩蓋「算得對不對」。
    let dev = Device::Cpu;
    // 準備一個確定性的隨機來源，原因見下面 Rng 的說明。
    let mut rng = Rng::new(42);

    // 造一個 768×768 的常態分佈矩陣。768 不是隨便挑的：
    // 它正好是你的 rwkv7-g1d-0.1b 的 hidden_size，也是 Q4_0（32）與 Q4_K（256）的公倍數。
    let shape = (768, 768);
    // 平均值 0、標準差 1，模擬訓練良好的權重（真實權重也接近這個形狀）。
    // 逐個抽樣後再包成張量——這樣才能保證每次執行的數字完全一致。
    let mut values = Vec::with_capacity(shape.0 * shape.1);
    for _ in 0..shape.0 * shape.1 {
        values.push(rng.next_normal());
    }
    let w = Tensor::from_vec(values, shape, &dev)?;
    // 元素總數，後面算壓縮率要用。
    let n = w.elem_count();

    println!("原始張量：{shape:?}，共 {n} 個元素");
    println!(
        "以 F32 存放需要 {}（每個權重 32 bit）\n",
        human_bytes(n * 4)
    );

    // 表頭。欄位順序對應下面每一列。
    println!(
        "{:<6} {:>10} {:>8} {:>8} {:>10} {:>10}",
        "型別", "大小", "位/權重", "壓縮比", "RMSE", "餘弦相似度"
    );
    println!("{}", "-".repeat(60));

    // 逐一走過對照表的每一種型別。
    for (dtype, name, block, type_size, _gguf_id) in dtype_table() {
        // 量化：把 f32 張量壓成 GGUF 區塊格式。
        // 注意它內部會先 flatten_all()——形狀只拿來檢查「能不能被 block 整除」，
        // 真正的量化是沿著攤平後的一維陣列切區塊，所以 Q4_0 沒有「每列一個縮放」的概念。
        let qt = QTensor::quantize(&w, dtype)?;

        // 還原回 f32，才能跟原值逐元素比較。
        let back = qt.dequantize(&dev)?;

        // 差值張量：量化誤差。
        let diff = (&w - &back)?;
        // 均方根誤差：把誤差平方取平均再開根號，單位跟原值一致，最好讀。
        let rmse = (diff.sqr()?.mean_all()?.to_scalar::<f32>()?).sqrt();
        // 最大絕對誤差：看最壞情況，量化爆掉時這個數字會突然變很大。
        let max_err = diff.abs()?.max_all()?.to_scalar::<f32>()?;

        // 餘弦相似度：方向對不對。量化只會讓向量「歪一點」，不會反向，
        // 所以這個值應該永遠接近 1；掉到 0.9 以下就代表量化率太激進了。
        let dot = (&w * &back)?.sum_all()?.to_scalar::<f32>()?;
        // 分母是兩個向量的 L2 範數相乘。
        let norm_w = w.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let norm_b = back.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let cosine = dot / (norm_w * norm_b);

        // 實際佔用位元組。這個值是「區塊數 × 每區塊位元組」，不是元素數 × 位元。
        let bytes = qt.storage_size_in_bytes();

        println!(
            "{:<6} {:>10} {:>8.2} {:>7.1}x {:>10.5} {:>10.6}",
            name,
            human_bytes(bytes),
            bits_per_weight(block, type_size),
            // 壓縮比 = F32 大小 ÷ 量化後大小。
            (n * 4) as f64 / bytes as f64,
            rmse,
            cosine
        );

        // 只印 Q8_0 的最大誤差當代表，其餘型別不洗版。
        if dtype == GgmlDType::Q8_0 {
            println!("       └ Q8_0 最大絕對誤差 = {max_err:.5}");
        }
    }

    // ── 第二段：離群值會發生什麼事 ────────────────────────────────────────
    println!("\n【離群值實驗】把第 0 個元素放大 100 倍，再量化一次");
    println!("  重點不是「哪種格式比較好」，而是「污染半徑有多大」。");

    // 複製一份，避免污染上面用過的張量。
    let mut raw: Vec<f32> = w.flatten_all()?.to_vec1()?;
    // 人為製造一個離群值。真實模型裡這種東西通常出現在 attention 的某些通道，
    // 或是 LayerNorm 之後的殘差上。
    raw[0] = 100.0;
    // 重新包成張量。
    let w2 = Tensor::from_vec(raw, shape, &dev)?;
    // 攤平後的原始值，方便逐區塊比對。
    let flat: Vec<f32> = w2.flatten_all()?.to_vec1()?;

    // 先算一個「對照組」：整張張量共用一個縮放係數的 4-bit 量化。
    // candle 沒有提供這種型別（GGUF 規格裡也沒有），但它正是「沒有區塊」的世界長什麼樣，
    // 自己寫 10 行就能看見差別有多大。
    let per_tensor = {
        // 找最大絕對值當縮放基準。4-bit 有號數的範圍是 -8..=7，所以除以 7。
        let max_abs = flat.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        // 步進大小。
        let scale = max_abs / 7.0;
        // 每個值：除以步進 → 四捨五入 → 夾在 4-bit 範圍內 → 乘回步進。
        flat.iter()
            .map(|v| ((v / scale).round().clamp(-8.0, 7.0)) * scale)
            .collect::<Vec<f32>>()
    };

    // 誤差要分兩區看，因為它們回答不同的問題：
    //   A. 離群值自己那 32 個權重 —— 誰能救得回來？
    //   B. 其餘 589792 個正常權重 —— 有沒有被牽連？
    // 只看 A 會得到「大家都一樣爛」的結論，而那個結論是錯的。
    let rmse_of = |a: &[f32], b: &[f32], from: usize, to: usize| -> f64 {
        // 平方誤差累加，用 f64 避免 100² 量級的精度損失。
        let mut acc = 0f64;
        for i in from..to {
            let d = a[i] as f64 - b[i] as f64;
            acc += d * d;
        }
        // 平均後開根號。
        (acc / (to - from) as f64).sqrt()
    };

    // 對照組的兩個數字。
    let pt_bad = rmse_of(&flat, &per_tensor, 0, 32);
    let pt_rest = rmse_of(&flat, &per_tensor, 32, flat.len());
    println!(
        "  per-tensor 4bit  離群區塊 {pt_bad:>8.4}   其餘權重 {pt_rest:>8.4}   ← 整張表共用一個 scale"
    );

    // 再跟真正的 GGUF 格式比。
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q8_0] {
        // 同樣的量化流程。
        let qt = QTensor::quantize(&w2, dtype)?;
        let back = qt.dequantize(&dev)?;
        // 攤平後的還原值。
        let flat_back: Vec<f32> = back.flatten_all()?.to_vec1()?;
        // 整體 RMSE：589824 個元素平均，離群值只佔 1 個，稀釋得很嚴重。
        let diff = (&w2 - &back)?;
        let rmse = (diff.sqr()?.mean_all()?.to_scalar::<f32>()?).sqrt();
        // 分區誤差。
        let bad = rmse_of(&flat, &flat_back, 0, 32);
        let rest = rmse_of(&flat, &flat_back, 32, flat.len());

        // 型別名先轉成字串，才能在 format 裡控制欄寬（`{:?}` 不能直接接寬度）。
        let dtype_name = format!("{dtype:?}");
        println!(
            "  {dtype_name:<5} 區塊量化     離群區塊 {bad:>8.4}   其餘權重 {rest:>8.4}   整體 {rmse:.5}   {}",
            human_bytes(qt.storage_size_in_bytes())
        );
    }

    println!("\n結論（以下每一句都對應上面的實測數字）：");
    println!("  1. 看「離群區塊」那一欄：per-tensor 與 Q4_0 都是 0.8996，一位不差。");
    println!("     這不是巧合——離群值就在第 0 個位置，它的區塊縮放係數必然等於");
    println!("     全域最大絕對值 ÷ 7，兩種做法在那個區塊上是同一個算式。");
    println!("     Q4_K 是 0.8040，好 11%，但仍在同一個量級——那 32 個權重基本上毀了。");
    println!("     **4-bit 救不了離群值本身，換哪種格式都只是好一點點。**");
    println!("  2. 看「其餘權重」那一欄：per-tensor 是 0.9992，Q4_0 是 0.0859，差 11.6 倍。");
    println!("     0.9992 這個數字很有戲：它幾乎等於原始權重的標準差（我們造的是 N(0,1)）。");
    println!("     意思是「正常權重全部被還原成 0」——步進 100/7≈14.3，比 99% 的權重都大，");
    println!("     四捨五入後全部歸零。這才是離群值最致命的傷害：不是它自己失真，");
    println!("     是它把整張表其他 58 萬個權重一起抹平。");
    println!("  3. 所以區塊量化的真正價值是「限制污染半徑」：從整個張量縮到 32 個權重。");
    println!("     以 0.1B 的 ffn.value（3072×768 = 235 萬個權重）來說，受災比例是 0.0014%。");
    println!("  4. Q4_K 在「其餘權重」上比 Q4_0 好 13%（0.0745 vs 0.0859），");
    println!("     靠的是 8 個子區塊各自 6-bit 縮放，讓正常權重的解析度更細。");
    println!("     它在離群區塊上也好 11%（0.8040 vs 0.8996），但那是「從毀掉變成毀得輕一點」，");
    println!("     不是修好——別把這 11% 誤讀成「Q4_K 能扛離群值」。");
    println!("  5. 只有 Q8_0 把離群區塊壓到 0.2408，代價是每權重 8.5 bit。");
    println!("     所以真正有效的策略不是換格式，而是「把離群值找出來、單獨處理」。");
    println!("  6. 這就是 llama.cpp 用 imatrix（重要性矩陣）加權的動機：");
    println!("     對「重要通道」給更高解析度，等效於把位元花在刀口上。");
    println!("     DeepSeek 在 MoE 上做 expert 級別的重要性統計，思路是同一條。");
    println!("  7. 實務結論：量化前先算每個張量的 max/abs-mean 比值，比值特別大的");
    println!("     （通常出現在 emb 與 ffn.value），就不要用 Q4_0，改 Q4_K 或直接留 Q8_0。");

    // 順手把 DType 印出來，提醒讀者：QTensor 不是 Tensor，兩者的 dtype 系統完全不同。
    println!("\n（對照）普通 Tensor 的 dtype = {:?}；QTensor 的 dtype = {:?}", DType::F32, GgmlDType::Q4K);

    Ok(())
}
