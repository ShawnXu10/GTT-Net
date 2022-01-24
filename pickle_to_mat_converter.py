import numpy, scipy.io
import pickle, sys

def main(source_name, dest_name):

	a=pickle.load( open( source_name, "rb" ) )


	scipy.io.savemat(dest_name, mdict={'pickle_data': a})

	print("Data successfully converted to .mat file with variable name \"pickle_data\"")
	

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])