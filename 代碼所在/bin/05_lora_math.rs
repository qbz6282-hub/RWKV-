//! 第 05 章：LoRA 的數學，以及它在 candle 裡為什麼必須「改模型原始碼」。
//!
//! 這支程式不載入 0.1B，只用一個 768×768 的線性層，
//! 3 秒就能看到「r=4 學得起來、r=2 學不起來」的差別。
//! 先在小尺度上把數學看透，再去動真模型，才不會把「調不動」誤判成「程式有 bug」。
//!
//! 執行：cargo run --release --bin 05_lora_math

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::loss;

/// 一個「凍結主權重 + 可訓練低秩旁路」的線性層。
///
/// 數學：`y = x·W₀ᵀ + (x·Aᵀ)·Bᵀ · (α/r)`
/// - `W₀` 是原始權重，**不是 Var**，所以不會有梯度、不會被優化器碰到。
/// - `A: (r, in)`、`B: (out, r)` 是新增的參數，只有這兩個會被訓練。
/// - `α/r` 是縮放係數：改 r 的時候輸出量級不會跟著變，超參數才好沿用。
///
/// 為什麼這樣就夠？因為微調時的權重變化 ΔW 本身是低秩的
/// （Aghajanyan 等人 2020 的實驗結果），所以用 rank-4~16 去逼近它，損失很小。
struct LoraLinear {
    /// 凍結的主權重，形狀 (out, in)。刻意用 Tensor 而不是 Var。
    base: Tensor,
    /// 低秩分解的下投影，形狀 (r, in)。
    a: Var,
    /// 低秩分解的上投影，形狀 (out, r)。
    b: Var,
    /// 縮放係數 α/r。
    scale: f64,
}

impl LoraLinear {
    /// 建構。`w0` 是原始權重，`r` 是秩，`alpha` 是縮放常數。
    fn new(w0: Tensor, r: usize, alpha: f64, dev: &Device) -> Result<Self> {
        // 取出形狀。注意 PyTorch 的 Linear 權重是 (out, in)。
        let (out_dim, in_dim) = w0.dims2()?;
        // A 用常態分佈初始化，標準差 0.02 是 LoRA 論文給的值。
        let a = Var::from_tensor(&Tensor::randn(0f32, 0.02f32, (r, in_dim), dev)?)?;
        // B 初始化為 0。這樣初始時 ΔW = B·A = 0，模型行為與原模型完全一致——
        // 這是 LoRA 最重要的性質：接上去不會破壞原有能力。
        let b = Var::from_tensor(&Tensor::zeros((out_dim, r), DType::F32, dev)?)?;
        Ok(Self {
            base: w0,
            a,
            b,
            // α/r 的縮放。
            scale: alpha / r as f64,
        })
    }

    /// 前向：主路徑 + 低秩旁路。
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 主路徑：x·W₀ᵀ。x 是 (n, in)，base.t() 是 (in, out)，結果 (n, out)。
        // 因為 base 不是 Var，這條路徑完全不建反傳節點，省記憶體。
        let main = x.matmul(&self.base.t()?)?;
        // 旁路第一段：x·Aᵀ → (n, r)。把 in 維壓到 r 維。
        let low = x.matmul(&self.a.as_tensor().t()?)?;
        // 旁路第二段：(n, r)·Bᵀ → (n, out)。再升回 out 維。
        let delta = low.matmul(&self.b.as_tensor().t()?)?;
        // 乘上縮放係數後與主路徑相加。
        // 這裡的加法是唯一的「融合點」，也是為什麼 LoRA 推論時可以把 B·A 併回 W₀。
        // 外面多包一層 Ok(...)，因為 Tensor 的運算回傳的是 candle::Result，
        // 而這個函式回傳 anyhow::Result，得靠 ? 做錯誤型別轉換。
        Ok(main.add(&(delta * self.scale)?)?)
    }

    /// 可訓練參數清單，交給優化器。
    fn trainable(&self) -> Vec<Var> {
        // 只有 A 和 B。
        vec![self.a.clone(), self.b.clone()]
    }

    /// 可訓練參數量。
    fn n_trainable(&self) -> usize {
        // A: r×in，B: out×r。
        self.a.as_tensor().elem_count() + self.b.as_tensor().elem_count()
    }

    /// 把旁路併回主權重，得到一個普通張量——這是部署時做的事。
    fn merge(&self) -> Result<Tensor> {
        // ΔW = (B·A) · scale，形狀 (out, in)。
        let delta = self.b.as_tensor().matmul(self.a.as_tensor())?;
        // 加回主權重。
        // 同樣要包一層 Ok(...) 做錯誤型別轉換。
        Ok(self.base.add(&(delta * self.scale)?)?)
    }
}

fn main() -> Result<()> {
    // CPU 就夠，這個實驗只有 60 萬參數。
    let dev = Device::Cpu;
    // 固定維度，跟 rwkv7-0.1b 的 receptance 投影一樣大。
    let (out_dim, in_dim) = (768usize, 768usize);

    println!("實驗設定：{out_dim}×{in_dim} 的線性層（等於 0.1B 的 receptance.weight）\n");

    // 原始權重 W₀，凍結。
    let w0 = Tensor::randn(0f32, 0.02f32, (out_dim, in_dim), &dev)?;
    // 人為造一個「真實的微調增量」ΔW，秩固定為 4。
    // 這是關鍵：我們知道答案的秩，所以能驗證 LoRA 的 rank 選對會怎樣、選錯會怎樣。
    let true_rank = 4;
    let u = Tensor::randn(0f32, 0.05f32, (out_dim, true_rank), &dev)?;
    let v = Tensor::randn(0f32, 0.05f32, (true_rank, in_dim), &dev)?;
    let delta_true = u.matmul(&v)?;
    // 目標權重 = W₀ + ΔW。
    let w_target = (&w0 + &delta_true)?;

    // 固定的輸入資料，用來產生「正確答案」。
    let x = Tensor::randn(0f32, 1f32, (64, in_dim), &dev)?;
    // y = x·W_targetᵀ。
    let y = x.matmul(&w_target.t()?)?;

    // 逐一測試不同的 rank。
    for r in [2usize, 4, 8, 32] {
        // 建 LoRA 層。
        let layer = LoraLinear::new(w0.clone(), r, 8.0, &dev)?;
        // 可訓練參數量。
        let n = layer.n_trainable();
        // 佔全參數的比例。
        let ratio = n as f64 * 100.0 / (out_dim * in_dim) as f64;

        // 優化器。LoRA 可以用比全量微調大很多的學習率，因為只動低秩方向。
        let mut opt = AdamW::new(
            layer.trainable(),
            ParamsAdamW {
                lr: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )?;

        // 訓練 300 步。
        let mut last = 0f32;
        for step in 0..300 {
            // 前向。
            let pred = layer.forward(&x)?;
            // 均方誤差。
            let loss = loss::mse(&pred, &y)?;
            // 每 50 步記錄一次。
            if step % 50 == 0 || step == 299 {
                last = loss.to_scalar::<f32>()?;
            }
            // 反傳 + 更新。backward_step 是 Optimizer trait 提供的便利方法。
            opt.backward_step(&loss)?;
        }

        println!(
            "r={:<3} 可訓練參數 {:>7}（{:>5.2}%）  最終 MSE = {:.3e}",
            r, n, ratio, last
        );

        // r 足夠時，順便驗證「合併後」的誤差。
        if r >= true_rank {
            // 合併成普通張量。
            let merged = layer.merge()?;
            // 與目標權重的最大絕對差。
            let max_diff = (&merged - &w_target)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            println!("      合併後與目標權重的最大差異 = {max_diff:.2e}");
        }
    }

    println!("\n【結論】");
    println!("  1. r < 真實秩 → 學不動，MSE 卡住。r ≥ 真實秩 → 幾乎精確收斂。");
    println!("     實務上我們不知道真實秩，所以業界慣例是 r=8~16（Qwen / Llama 微調社群的值），");
    println!("     再用驗證集 loss 往上或往下調。");
    println!("  2. 參數量是 r×(in+out)，跟 in×out 相比小兩三個數量級——");
    println!("     768×768 在 r=4 時只有 1.04%，這就是 6 GB 顯存能微調 1.5B 的原因。");
    println!("  3. merge() 之後完全沒有推論開銷：B·A 併回 W₀，前向跟原模型一模一樣。");

    // ── 為什麼在 candle 裡 LoRA 特別麻煩 ────────────────────────────────
    println!("\n【candle 的結構限制】");
    println!("  PyTorch 可以 `replace_module(model, 'att.receptance', lora_layer)` 就地換掉一層；");
    println!("  candle 沒有這種反射機制。翻開 candle-transformers 的 rwkv_v7.rs 你會看到：");
    println!();
    println!("      pub struct Model {{");
    println!("          embeddings: Embedding,");
    println!("          blocks: Vec<Block>,");
    println!("          ln_out_weight: Tensor,   // ← 是 Tensor，不是 Var");
    println!("          head_t: Tensor,");
    println!("      }}");
    println!();
    println!("  而且載入時就做了 `.t()?.contiguous()`，權重被固定成特定佈局。");
    println!("  所以要用 LoRA，只有兩條路：");
    println!("    A. fork rwkv_v7.rs，把 att/ffn 的欄位型別從 Tensor 換成 LoraLinear。");
    println!("       改動點共 6 處（receptance / key / value / output / ffn.key / ffn.value），");
    println!("       我在文件二第 7 節列了完整清單。");
    println!("    B. 用「凍結式微調」代替：全部參數都是 Var，但只把少數幾層交給優化器。");
    println!("       這不是 LoRA（沒有低秩約束），但一行都不用改模型，適合先驗證資料有沒有問題。");
    println!();
    println!("  這也是為什麼 mistral.rs 這種專案要自己重寫一份模型——不是為了炫技，");
    println!("  是因為要做 ISQ 與 LoRA，就必須在模型裡留下「可插拔」的縫。");

    Ok(())
}
