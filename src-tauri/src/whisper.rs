use crate::store::STORE;
use libc::c_void;
use std::ffi::CStr;
use std::path::PathBuf;
use tauri::Manager;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

#[derive(Clone, serde::Serialize, Debug)]
struct WhisperPayload {
    status: String,
    message: String,
}

unsafe extern "C" fn whisper_callback(
    _: *mut whisper_rs_sys::whisper_context,
    ptr: *mut whisper_rs_sys::whisper_state,
    _: i32,
    app: *mut c_void,
) {
    let i_segment = whisper_rs_sys::whisper_full_n_segments_from_state(ptr) - 1;
    let c_str_ptr = whisper_rs_sys::whisper_full_get_segment_text_from_state(ptr, i_segment);
    if c_str_ptr.is_null() {
        return;
    }
    let c_str = CStr::from_ptr(c_str_ptr);
    let subtitle = match c_str.to_str() {
        Ok(str) => str.to_owned(),
        Err(_) => {
            let app_handle = Box::from_raw(app as *mut tauri::AppHandle);
            let message = "Text segment could not be converted to string.".to_string();
            emit_err(&app_handle, &message);
            return;
        }
    };

    let app_handle = Box::from_raw(app as *mut tauri::AppHandle);
    STORE.lock().unwrap().push_data(
        &app_handle,
        whisper_rs_sys::whisper_full_get_segment_t0_from_state(ptr, i_segment) * 10,
        whisper_rs_sys::whisper_full_get_segment_t1_from_state(ptr, i_segment) * 10,
        subtitle,
    );
    let _ = Box::into_raw(app_handle);
}

pub async fn run(
    path_wav: &str,
    path_model: &str,
    lang: &str,
    translate: bool,
    offset_ms: i32,
    duration_ms: i32,
    app: &tauri::AppHandle,
) -> Result<(), String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let audio_data;
    let lang_string;
    let context;
    {
        let mut config = STORE.lock().map_err(|_| "Mutex is poisoned")?;
        // Storeの設定を更新する処理
        // ...

        let reader_result = hound::WavReader::open(config.get_path_wav());
        if reader_result.is_err() {
            emit_err(app, "指定されたwavファイルを開けませんでした");
            return Err("指定されたwavファイルを開けませんでした".to_string());
        }
        let mut reader = reader_result.unwrap();
        audio_data = reader
            .samples::<i16>()
            .map(|sample| sample.map(|s| s as f32 / i16::MAX as f32))
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|_| "Failed to read samples from WAV file".to_string())?;

        lang_string = config.get_lang().unwrap_or("ja").to_string();
        params.set_language(Some(&lang_string));
        params.set_translate(config.get_translate());
        params.set_offset_ms(config.get_ms_offset());
        params.set_duration_ms(config.get_ms_duration());
        params.set_tdrz_enable(true);
        params.set_suppress_non_speech_tokens(true);
        
        // コールバックとユーザーデータの設定
        unsafe {
            params.set_new_segment_callback(Some(whisper_callback));
            params.set_new_segment_callback_user_data(
                Box::into_raw(Box::new(app.clone())) as *mut c_void
            );
        }

        context = WhisperContext::new_with_params(
            config.get_path_model().to_str().unwrap(),
            WhisperContextParameters::default(),
        )
        .map_err(|_| "言語モデルの読み込みに失敗しました".to_string())?;
    }

    // エラーハンドリングを伴うStateの作成
    let mut state = context.create_state().map_err(|_| {
        emit_err(app, "Whisper Stateの初期化に失敗しました");
        "Whisper Stateの初期化に失敗しました".to_string()
    })?;
    
    // 開始イベントを送信
    if let Err(_) = app.emit_all(
        "whisper",
        WhisperPayload {
            status: "start".to_string(),
            message: "初期化が完了しました。文字起こしを開始します。".to_string(),
        },
    ) {
        return Err("イベントの送信に失敗しました".to_string());
    }

    // 文字起こし処理の実行
    state.full(params, &audio_data[..]).map_err(|_| {
        emit_err(app, "言語モデルの実行に失敗しました");
        "言語モデルの実行に失敗しました".to_string()
    })?;
    Ok(())
}

fn emit_err(app: &tauri::AppHandle, msg: &str) {
    let _ = app.emit_all(
        "whisper",
        WhisperPayload {
            status: "error".to_string(),
            message: msg.to_string(),
        },
    );
}
