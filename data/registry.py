
DATA_SOURCES = {
    # Fama-French
    "FF/SMB": {
        "type": "zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    },
    "FF/HML": {
        "type": "zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    },

    # Yahoo Finance
    "Yahoo/SP500": {
        "type": "yahoo",
        "ticker": "^GSPC"
    },

    # Placeholder for other stuff
    "Custom/Thing": {
        "type": "custom",
        "handler": "my_custom_loader"
    }
}