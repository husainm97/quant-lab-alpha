
DATA_SOURCES = {
    # Fama-French
    "FF/SMB": {
        "type": "zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_ME_TXT.zip"
    },
    "FF/HML": {
        "type": "zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_BE-ME_TXT.zip"
    },
    "FF/RMW": {
        "type": "zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_OP_TXT.zip"
    },
    "FF/CMA": {
        "type": "zip",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_INV_TXT.zip"
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