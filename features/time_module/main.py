import time

def show_current_time():
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return current_time

# Standard entrypoint for the agent system
def run(query: str = "") -> str:
    return f"The current system time is: {show_current_time()}"

if __name__ == "__main__":
    print(run())