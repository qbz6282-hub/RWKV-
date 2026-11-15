//! 共用工具：模型路徑、位元組格式化、量化型別對照表。
//!
//! 這裡只放「五個範例都要用」的東西，不放任何演算法，
//! 免得讀者要理解主線時還得先翻過一堆輔助函式。

use candle_core::quantized::GgmlDType;
use std::path::PathBuf;

/// 你的模型目錄。路徑寫死是刻意的：這份教學綁定你筆電上的實際檔案，
/// 換機器時只要改這一行，五個範例全部跟著生效。
pub fn models_dir() -> PathBuf {
    PathBuf::from(r"C:\Users\acer\RustroverProjects\RWKV_shi2jian4\models")
}

/// 0.1B（g1d 世代）：768 隱藏維度、12 層、BF16、382 MB。
/// 這是本教學的主角——量化快、微調塞得進 6 GB 顯存。
pub fn small_01b() -> PathBuf {
    models_dir().join("rwkv7-g1d-0.1b-20260129-ctx8192.safetensors")
}

/// 1.5B（g1d 世代）：2048 隱藏維度、24 層、BF16、3.05 GB。
/// 只拿來做「同一個流程換成大模型會怎樣」的對照。
pub fn mid_15b() -> PathBuf {
    models_dir().join("rwkv7-g1d-1.5b-20260212-ctx8192.safetensors")
}

/// 2.9B（g1k 世代）：2560 隱藏維度、32 層、已經是 Q8_0 的 GGUF、3.26 GB。
/// 這是「別人量化好的成品」，用來學怎麼讀 GGUF、以及對照 candle 缺什麼。
pub fn gguf_29b() -> PathBuf {
    models_dir().join("rwkv7-g1k-2.9b-Q8_0.gguf")
}

/// RWKV World 分詞器表（65536 個詞元）。注意它沒有 encode 方法，只有 decode，
/// 原因見第二章「為什麼我們用位元組當分詞器」。
pub fn vocab_json() -> PathBuf {
    models_dir().join("rwkv_vocab_v20230424.json")
}

/// 把位元組數轉成人看的字串。量化章節從頭到尾都在比大小，這個函式會一直出現。
pub fn human_bytes(n: usize) -> String {
    // 單位階梯，從位元組到 TiB。量化討論不會超過 TiB，但寫滿比較安心。
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    // 用 f64 是因為中間值會有小數，例如 382.17 MiB。
    let mut value = n as f64;
    // 目前走到第幾階。
    let mut idx = 0;
    // 每超過 1024 就往上跳一階，但最多跳到最後一階，避免索引越界。
    while value >= 1024.0 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    // 位元組階段不帶小數（「382 B」比「382.00 B」自然），其他階段帶兩位。
    if idx == 0 {
        format!("{n} {}", UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[idx])
    }
}

/// 量化型別的完整對照表：(型別, 簡稱, 每區塊元素數, 每區塊位元組數, GGUF 型別編號)。
///
/// 這張表是整份教學的地基，值得背下來三件事：
/// - **block_size** 決定「幾個數值共用一組縮放係數」，越小越精確、越大越省。
/// - **type_size / block_size** 就是每個數值實際佔幾個位元組，例如 Q4_0 是 18/32 ≈ 0.5625。
/// - **GGUF 型別編號** 是寫檔時寫進 header 的 u32，llama.cpp 靠它認型別；
///   candle 的 `to_u32()` 是 `pub(crate)`，所以我們自己維護一份。
pub fn dtype_table() -> Vec<(GgmlDType, &'static str, usize, usize, u32)> {
    vec![
        // 純浮點：block_size 都是 1，等於「沒有量化」，只是換容器。
        (GgmlDType::F32, "F32", 1, 4, 0),
        (GgmlDType::F16, "F16", 1, 2, 1),
        // BF16 在 GGUF 規格裡是編號 30（不是 2），這個編號很反直覺，容易寫錯。
        (GgmlDType::BF16, "BF16", 1, 2, 30),
        // 舊式（Q4_0 家族）：32 個數值一組，只有一個 f16 縮放係數。
        (GgmlDType::Q4_0, "Q4_0", 32, 18, 2),
        (GgmlDType::Q4_1, "Q4_1", 32, 20, 3),
        (GgmlDType::Q5_0, "Q5_0", 32, 22, 6),
        (GgmlDType::Q5_1, "Q5_1", 32, 24, 7),
        (GgmlDType::Q8_0, "Q8_0", 32, 34, 8),
        (GgmlDType::Q8_1, "Q8_1", 32, 36, 9),
        // K-quant 家族：256 個數值一個 super-block，內部再切 8 個子區塊各自縮放。
        // 多一層縮放 = 同一個位元數下誤差更小，代價是解量化程式碼複雜得多。
        (GgmlDType::Q2K, "Q2_K", 256, 84, 10),
        (GgmlDType::Q3K, "Q3_K", 256, 110, 11),
        (GgmlDType::Q4K, "Q4_K", 256, 144, 12),
        (GgmlDType::Q5K, "Q5_K", 256, 176, 13),
        (GgmlDType::Q6K, "Q6_K", 256, 210, 14),
        (GgmlDType::Q8K, "Q8_K", 256, 292, 15),
    ]
}

/// 每個數值平均佔幾個位元（bit）。這是比較量化率的唯一公平指標——
/// 直接比檔案大小會被「有沒有把 1D 參數也量化」干擾。
pub fn bits_per_weight(block_size: usize, type_size: usize) -> f64 {
    // 一個區塊的位元數 ÷ 一個區塊的元素數 = 每權重位元數。
    (type_size * 8) as f64 / block_size as f64
}
