import os
import json
import subprocess

WORKSPACE = "./workspace"
MEMORY = []

os.makedirs(WORKSPACE, exist_ok=True)

def read_file(filename):
    path = os.path.join(WORKSPACE, filename)

    try:
        with open(path,"r",encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return str(e)

def write_file(filename, content):
    path = os.path.join(WORKSPACE, filename)

    with open(path,"w",encoding="utf-8") as f:
        f.write(content)

    return "寫入成功"

def run_command(cmd):

    whitelist = [
        "ls",
        "dir",
        "pwd",
        "echo"
    ]

    if cmd.split()[0] not in whitelist:
        return "指令不允許"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        return result.stdout

    except Exception as e:
        return str(e)

def tool_router(tool):

    name = tool["tool"]

    if name == "read_file":
        return read_file(tool["file"])

    elif name == "write_file":
        return write_file(
            tool["file"],
            tool["content"]
        )

    elif name == "shell":
        return run_command(tool["command"])

    return "未知工具"

def main():

    print("AgentLite 啟動")

    while True:

        user = input("你：")

        if user == "exit":
            break

        MEMORY.append(user)

        print("請輸入 JSON Tool Call")

        tool_text = input("> ")

        try:
            tool = json.loads(tool_text)

            result = tool_router(tool)

            print("結果：")
            print(result)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()