import subprocess

for x in range(20):
    print(x)
    subprocess.run("python ntd_person.py", shell=True)
