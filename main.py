import subprocess

def start_program():
 
    command = ["lama-cleaner", "--device=cpu", "--port=8080"]

    try:
    
        subprocess.run(command)
    except FileNotFoundError:
        print("Error")

if __name__ == "__main__":
    start_program()
