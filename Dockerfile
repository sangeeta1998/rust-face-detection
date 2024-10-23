# 1. Stage: Build the Rust project
FROM rust:1.71 as builder

# Set the working directory inside the container
WORKDIR /usr/src/face-detection

# Copy the Cargo.toml and lock file to cache dependencies first
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to cache the dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build the project to cache dependencies
RUN cargo build --release

# Remove the dummy main.rs and copy the actual project files
RUN rm -rf src
COPY ./src ./src

# Build the actual project
RUN cargo build --release

# 2. Stage: Create the runtime environment
FROM debian:buster-slim

# Install necessary dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    libtensorflow1.15 \
    libopencv-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /usr/src/face-detection

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/face-detection/target/release/face-detection .

# Copy the input image and model files to the runtime container
COPY pexels-valeriya-1805164.png ./pexels-valeriya-1805164.png
COPY ../ssd_mobilenet_v2_coco_2018_03_29 ./ssd_mobilenet_v2_coco_2018_03_29

# Command to run the face detection program
CMD ["./face-detection"]

