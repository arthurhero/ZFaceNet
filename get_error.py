import subprocess

error_path = "tmp_invalid/"

def main():
    count=0

    cmd = "ls "+error_path
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    files = output.splitlines()
    for f in files:
        cmd = "cat "+error_path+f
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        entries = output.splitlines()
        for entry in entries:
            if entry == "error":
                count += 1
        print f + " " + str(count)
        count = 0

main()
