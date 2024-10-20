use std::io::Write;
use std::sync::LazyLock;
use axum::{
    routing::get,
    routing::post,
    Router
};
use axum_extra::extract::Multipart;
use BvrDetect::{self, bvr_detect};
use BvrDetect::data::{BvrImage, DeviceType, ModelConfig, ProcessingType, YoloVersion};
use clap::Parser;
use serde::Serialize;
use tokio::sync::Mutex;
use tokio::time::Instant;

static MODEL_CONFIG: LazyLock<Mutex<ModelConfig>> = LazyLock::new(|| Mutex::new({
    ModelConfig {
        weights_path: String::new(),
        ort_lib_path: String::new(),
        labels_path: String::new(),
        device_type: Default::default(),
        processing_type: Default::default(),
        yolo_version: Default::default(),
        conf_threshold: 0.3,
        width: 640,
        height: 640,
        split_wide_input: false,
    }
}));

/// Args parser
#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(about = "Object detection endpoint for applications to use")]
struct Cli {
    /// Path to ONNX file. eg ./yolov9m.onnx
    #[arg(short, long)]
    model_path: String,

    /// Path to classes file. eg ./coco.names or ./labels.txt
    #[arg(short, long)]
    classes_path: String,

    /// Path to onnx runtime library. eg ./libonnxruntime.so.1.19.0
    #[arg(short='l', long)]
    lib_path_ort: String,

    /// Device to use for inference. Options are [CPU, CUDA, TensorRT]
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Detection processing method to use. Either built-in onnxruntime or external python script. Options are [ort, torch, python]
    #[arg(short='r', long, default_value = "ort")]
    processing_type: String,

    /// Set Yolo version. Options are [v5, v6, v7, v8, v9, v10, v11]
    #[arg(short, long, default_value = "v11")]
    yolo_version: String,

    /// Split extra wide images to process each half individually
    #[arg[short, long, action]]
    split_wide: bool,

    /// Override detection threshold
    #[arg(short, long, default_value_t = 0.4)]
    threshold: f32,

    /// The model's input width
    #[arg(short, long, default_value_t = 640)]
    width: u32,

    /// The model's input height
    #[arg(short='e', long, default_value_t = 640)]
    height: u32,

    /// The port that should be used
    #[arg(short, long, default_value_t = 3000)]
    port: u16,

    /// Set logging level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Serialize)]
struct Results {
    success: bool,
    #[serde(rename="@inferenceMs")]
    inference_ms: u32,
    count: u32,
    predictions: Vec<BasicDetectionResults>
}

#[derive(Debug, Serialize)]
struct BasicDetectionResults {
    label: String,
    confidence: f32,
    x_min: i32,
    x_max: i32,
    y_min: i32,
    y_max: i32,
}

/// Main function that sets up the object detection API server using Axum.
/// It parses command-line arguments, configures the model settings, and starts the server
/// to listen for requests on the specified port.
#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.log_level.to_lowercase().as_str() {
        "trace" => simple_logger::init_with_level(log::Level::Trace).unwrap(),
        "debug" => simple_logger::init_with_level(log::Level::Debug).unwrap(),
        "warn" => simple_logger::init_with_level(log::Level::Warn).unwrap(),
        "error" => simple_logger::init_with_level(log::Level::Error).unwrap(),
        _ => simple_logger::init_with_level(log::Level::Info).unwrap(),
    }

    let mut model_config = MODEL_CONFIG.lock().await;

    model_config.weights_path = cli.model_path.to_string();
    model_config.labels_path = cli.classes_path.to_string();
    model_config.ort_lib_path = cli.lib_path_ort.to_string();
    model_config.yolo_version = YoloVersion::from(cli.yolo_version);
    model_config.conf_threshold = if cli.threshold > 1.0 { cli.threshold / 100.0 } else { cli.threshold };
    model_config.split_wide_input = cli.split_wide;
    model_config.width = cli.width;
    model_config.height = cli.height;

    match ProcessingType::from_str(cli.processing_type.as_str()) {
        Some(ProcessingType::ORT) => model_config.processing_type = ProcessingType::ORT,
        Some(ProcessingType::Python) => model_config.processing_type = ProcessingType::Python,
        _ => {
            println!("Unknown choice for processing type: {}\nDefaulting to 'Native'", cli.processing_type);
            model_config.processing_type = ProcessingType::ORT
        }
    }

    match DeviceType::from_str(cli.device.as_str(), 0) {
        Some(DeviceType::CPU) => model_config.device_type = DeviceType::CPU,
        Some(DeviceType::CUDA(0)) => model_config.device_type = DeviceType::CUDA(0),
        Some(DeviceType::TensorRT(0)) => model_config.device_type = DeviceType::TensorRT(0),
        _ => {
            println!("Unknown device for execution provider: {}\nDefaulting to 'CPU'", cli.device);
            model_config.device_type = DeviceType::CPU
        }
    }

    println!("\nUsing the following config:\n{}", model_config.to_string());

    drop(model_config);

    // setup routes
    let app = Router::new()
        .route("/", get(root))
        .route("/detect", post(detect))
        .route("/v1/vision/detection", post(v1_detect)); // Compatibility endpoint

    let bind_address = format!("0.0.0.0:{}", &cli.port);

    let listener = tokio::net::TcpListener::bind(bind_address).await.unwrap();

    println!("\nListening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app).await.unwrap();
}

/// Root endpoint handler that returns the status of the web API and detection process.
///
/// # Returns
/// * `String` - A message indicating if the web API is running and if the detection process is initialized.
async fn root() -> String {
    let is_running = bvr_detect::is_running().await.unwrap();
    let res = format!("Web api: Running\nDetection process initialized: {}", is_running);
    res
}

/// Compatibility endpoint handler for v1 API calls. This function forwards
/// the request body to the `detect` function for processing.
///
/// # Arguments
/// * `body` - The binary data of the image to be processed in the detection.
///
/// # Returns
/// * `String` - A JSON string representing the detected objects and their bounding boxes.
async fn v1_detect(multipart: Multipart) -> String {
    detect(multipart).await
}

/// Object detection handler that processes the image in the request body and returns detection results.
///
/// # Arguments
/// * `body` - The binary image data submitted via a POST request.
///
/// # Returns
/// * `String` - A JSON string containing a list of detected objects, their labels, confidence scores, and bounding boxes.
async fn detect(mut multipart: Multipart) -> String {
    let mut min_confidence = 0.0;
    let mut image_data = Vec::new();
    let mut file_found = false;

    let now = Instant::now();
    let model_config = MODEL_CONFIG.lock().await;
    println!("\n{} - Running detection", chrono::offset::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string());

    // Iterate over the multipart fields
    while let Some(mut field) = multipart.next_field().await.unwrap() {
        let field_name = &field.name().unwrap_or_default().to_string();

        if field_name == "min_confidence" {
            let confidence_value = field.text().await.unwrap_or_else(|_| "0.0".to_string());
            min_confidence = confidence_value.parse::<f32>().unwrap_or(0.0);

            // if provided threshold value is above 1.0, we assume it's a 0 - 100 int
            // value and convert to f32 0.0 - 1.0
            min_confidence = if min_confidence > 1.0 { min_confidence / 100.0 } else { min_confidence };
        }
        else if field_name == "image" {
            let filename = field.file_name().unwrap_or("unknown").to_string();

            while let Ok(Some(chunk)) = field.chunk().await {
                if let Err(e) = image_data.write_all(&chunk) {
                    println!("Error writing file chunk: {:?}", e);
                    return "Failed to parse image".to_string()
                }
            }

            if let Err(e) = field.chunk().await {
                // Handle the case where `field.chunk()` returns an error
                println!("Error reading file chunk: {:?}", e);
                return "Failed to read image".to_string()
            }

            file_found = true;
            println!("Received file: {}", filename);
        }
    }

    if file_found {
        let image = image::load_from_memory(&image_data).map_err(|err| {
            println!("Image decoding failed: {:?}", err);
        }).unwrap();

        let img_width = image.width() as i32;
        let img_height = image.height() as i32;

        let bvr_image = BvrImage {
            image,
            img_width,
            img_height,
            threshold: min_confidence,
            augment: false,
        };

        if !bvr_detect::is_running().await.unwrap() {
            bvr_detect::init_detector(model_config.clone(), false).await;
        }

        let detections = bvr_detect::detect(bvr_image).await.expect("Error processing detection");
        let mut detection_time_str = "(no results)".to_string();

        if detections.is_empty() {
            let empty_results = Results {
                success: true,
                inference_ms: 0,
                count: 0,
                predictions: Vec::new(),
            };

            println!("Found ZERO objects");
            // Return basic info
            return serde_json::to_string(&empty_results).unwrap()
        };

        let detection_time = detections[0].last_inference_time / 1000;
        if detections.len() > 0 {
            detection_time_str = detection_time.to_string() + "ms";
        }

        let mut prediction_results: Vec<BasicDetectionResults> = Vec::new();

        for detect in detections {
            let (x1, y1, x2, y2) = detect.bbox.as_x1y1_x2y2_i32();
            let bi_result = BasicDetectionResults {
                label: detect.label.unwrap_or("unknown".to_string()),
                confidence: detect.confidence,
                x_min: x1,
                x_max: x2,
                y_min: y1,
                y_max: y2,
            };

            prediction_results.push(bi_result);
        }

        let results = Results {
            success: true,
            inference_ms: detection_time as u32,
            count: prediction_results.len() as u32,
            predictions: prediction_results,
        };

        let results_str = serde_json::to_string(&results).unwrap();

        println!("Inference time: {}  |  Total process time: {}ms", detection_time_str, now.elapsed().as_millis());
        println!("Found {} objects", results.count);
        for det in results.predictions {
            println!("{}", serde_json::to_string(&det).unwrap());
        }

        results_str
    } else {
        "No file found in the request".to_string()
    }
}