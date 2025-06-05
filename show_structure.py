import os

def print_tree(startpath, indent=""):
    for item in sorted(os.listdir(startpath)):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}/")
            print_tree(path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

if __name__ == "__main__":
    folder_path = "."  # Change this to your project path if needed
    print(f"📦 Project structure of: {os.path.abspath(folder_path)}\n")
    print_tree(folder_path)
