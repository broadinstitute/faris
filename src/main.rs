fn main() {
    match faris::run() {
        Ok(_) => println!("Server shut down without error."),
        Err(e) => eprintln!("Server shut down with error: {e}"),
    }
}