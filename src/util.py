import subprocess, os

def parseline(mystr):
    ind = mystr.index('#')
    return mystr[ind+1:]

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

def mymkdir(mydir):
    if not os.path.exists(mydir):
        print('making ',mydir)
        os.mkdir(mydir)
