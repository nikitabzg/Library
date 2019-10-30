
"""
Verifies the received format
"""

def verify_format(data, labels):
    if data.shape[0] != labels.shape[0] : 
        raise AttributeError("Shape of the data is not equal to shape of the labels")
    print("Correct shape for data and labels")