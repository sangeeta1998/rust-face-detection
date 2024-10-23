use std::fs::File;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, Tensor};
use image::{DynamicImage, GenericImageView, Pixel};

fn main() {
    // Start measuring time as soon as the program starts
    let start_time = Instant::now();

    // Get current system time (for logging "Main function started at")
    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    // Log the start of the main function with a timestamp
    println!("Main function started at: {}", current_time);

    // Load the pre-trained model (frozen_inference_graph.pb)
    let model_path = "/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"; 
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    let mut file = File::open(Path::new(model_path)).unwrap();
    file.read_to_end(&mut proto).unwrap();
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new()).unwrap();

    // Initialize the session
    let session = Session::new(&SessionOptions::new(), &graph).unwrap();

    // Prepare input image
    let image_path = "pexels-valeriya-1805164.png";
    let image = image::open(image_path).unwrap();
    let (width, height) = image.dimensions();
    let input_tensor = Tensor::new(&[1, height as u64, width as u64, 3]);

    // Run the session
    let mut step = tensorflow::StepWithGraph::new();
    step.add_input(&graph.operation_by_name_required("image_tensor").unwrap(), 0, &input_tensor);
    step.add_target(&graph.operation_by_name_required("detection_boxes").unwrap());

    let output_tensor = step.take_output(0).unwrap();

    // Process output (bounding boxes, scores, etc.)
    println!("Face detection output: {:?}", output_tensor);

    // Measure total time and print
    let duration = start_time.elapsed();
    println!("Startup time: {:?}", duration);
}

