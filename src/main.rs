use std::io::Cursor;
use std::sync::LazyLock;
use axum::{
    routing::get,
    routing::post,
    Router,
    body::Bytes
};

use BvrDetect::{self, bvr_detect};
use BvrDetect::bvr_data;
use BvrDetect::bvr_data::{ProcessingType, DeviceType, ModelConfig};
use clap::Parser;
use image::ImageReader;
use serde::Serialize;
use tokio::sync::Mutex;
use tokio::time::Instant;

static MODEL_CONFIG: LazyLock<Mutex<ModelConfig>> = LazyLock::new(|| Mutex::new({
    ModelConfig {
        onnx_path: String::new(),
        ort_lib_path: String::new(),
        classes_path: String::new(),
        device_type: Default::default(),
        detector_type: Default::default(),
        width: 640,
        height: 640,
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

    /// Detection processing method to use. Either built-in onnxruntime or external python script. Options are [native, python]
    #[arg(short='t', long, default_value = "native")]
    processing_type: String,

    /// The model's input width
    #[arg(short, long, default_value_t = 640)]
    width: u32,

    /// The model's input height
    #[arg(short='e', long, default_value_t = 640)]
    height: u32,

    /// The port that should be used
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
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

    tracing_subscriber::fmt::init();

    let mut model_config = MODEL_CONFIG.lock().await;

    model_config.onnx_path = cli.model_path.to_string();
    model_config.classes_path = cli.classes_path.to_string();
    model_config.ort_lib_path = cli.lib_path_ort.to_string();
    model_config.width = cli.width;
    model_config.height = cli.height;

    match ProcessingType::from_str(cli.processing_type.as_str()) {
        Some(ProcessingType::Native) => model_config.detector_type = ProcessingType::Native,
        Some(ProcessingType::Python) => model_config.detector_type = ProcessingType::Python,
        _ => {
            println!("Unknown choice for processing type: {}\nDefaulting to 'Native'", cli.processing_type);
            model_config.detector_type = ProcessingType::Native
        }
    }

    match DeviceType::from_str(cli.device.as_str()) {
        Some(DeviceType::CPU) => model_config.device_type = DeviceType::CPU,
        Some(DeviceType::CUDA) => model_config.device_type = DeviceType::CUDA,
        Some(DeviceType::TensorRT) => model_config.device_type = DeviceType::TensorRT,
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
async fn v1_detect(body: Bytes) -> String {
    detect(body).await
}

/// Object detection handler that processes the image in the request body and returns detection results.
///
/// # Arguments
/// * `body` - The binary image data submitted via a POST request.
///
/// # Returns
/// * `String` - A JSON string containing a list of detected objects, their labels, confidence scores, and bounding boxes.

async fn detect(body: Bytes) -> String {
    let now = Instant::now();
    let model_config = MODEL_CONFIG.lock().await;
    println!("\n{} - Running detection", chrono::offset::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string());

    let byte_slice = &body[..];
    let cursor = Cursor::new(byte_slice);
    let image = ImageReader::new(cursor)
        .with_guessed_format().unwrap().decode().unwrap();

    let img_width = image.width() as i32;
    let img_height = image.height() as i32;

    let bvr_image = bvr_data::BvrImage {
        image,
        img_width,
        img_height,
        conf_thres: 0.7,
        iou_thres: 0.5,
        augment: false,
    };

    if !bvr_detect::is_running().await.unwrap() {
        let model_init = ModelConfig {
            onnx_path: model_config.onnx_path.clone(),
            ort_lib_path: model_config.ort_lib_path.clone(),
            classes_path: model_config.classes_path.clone(),
            device_type: model_config.device_type,
            detector_type: Default::default(),
            width: model_config.width,
            height: model_config.height,
        };
        bvr_detect::init_detector(model_init, false).await;
    }

    let detections = bvr_detect::detect(bvr_image).await.expect("Error processing detection");
    let mut detection_time = "(no results)".to_string();

    if detections.len() > 0 {
        detection_time = (detections[0].last_inference_time / 1000).to_string() + "ms";
    }

    let mut results: Vec<BasicDetectionResults> = Vec::new();

    for detect in detections {
        let bi_result = BasicDetectionResults {
            label: detect.label,
            confidence: detect.confidence,
            x_min: detect.bbox.x1,
            x_max: detect.bbox.x2,
            y_min: detect.bbox.y1,
            y_max: detect.bbox.y2,
        };

        results.push(bi_result);
    }

    let results_str = serde_json::to_string(&results).unwrap();

    println!("Inference time: {}  |  Total process time: {}ms", detection_time, now.elapsed().as_millis());
    println!("Found {} objects", results.len());
    for det in results {
        println!("{}", serde_json::to_string(&det).unwrap());
    }

    results_str
}