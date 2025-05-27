fn main() {
    match faris::run() {
        Ok(_) => println!("Execution completed successfully."),
        Err(e) => eprintln!("An error occurred: {}", e),
    }
}