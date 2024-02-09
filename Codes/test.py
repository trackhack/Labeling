import argparse
import time

# Create an argument parser
parser = argparse.ArgumentParser()

# Add a flag argument named '-f' or '--flag'
parser.add_argument('-f', '--flag', action='store', help='Specify the flag value')

# Parse the command line arguments
args = parser.parse_args()

time.sleep(5)
# Check if the flag was provided
if args.flag:
    print("Flag:", args.flag)
else:
    print("No flag specified.")
