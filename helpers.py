def country_to_country_code(country: str) -> str:
    """
    Returns the standard 3-letter FIFA country code for the 48 teams
    participating in the 2026 FIFA World Cup.
    """
    mapping = {
        # Group A
        "mexico": "MEX",
        "south africa": "RSA",
        "south korea": "KOR",
        "republic of korea": "KOR",
        "korea republic": "KOR",
        "czech republic": "CZE",
        "czechia": "CZE",
        # Group B
        "canada": "CAN",
        "bosnia and herzegovina": "BIH",
        "bosnia": "BIH",
        "qatar": "QAT",
        "switzerland": "SUI",
        # Group C
        "brazil": "BRA",
        "morocco": "MAR",
        "haiti": "HAI",
        "scotland": "SCO",
        # Group D
        "united states": "USA",
        "usa": "USA",
        "united states of america": "USA",
        "paraguay": "PAR",
        "australia": "AUS",
        "turkey": "TUR",
        "türkiye": "TUR",
        # Group E
        "germany": "GER",
        "curaçao": "CUW",
        "curacao": "CUW",
        "ivory coast": "CIV",
        "côte d'ivoire": "CIV",
        "cote d'ivoire": "CIV",
        "ecuador": "ECU",
        # Group F
        "netherlands": "NED",
        "japan": "JPN",
        "sweden": "SWE",
        "tunisia": "TUN",
        # Group G
        "belgium": "BEL",
        "egypt": "EGY",
        "iran": "IRN",
        "ir iran": "IRN",
        "new zealand": "NZL",
        # Group H
        "spain": "ESP",
        "cape verde": "CPV",
        "cabo verde": "CPV",
        "saudi arabia": "KSA",
        "uruguay": "URU",
        # Group I
        "france": "FRA",
        "senegal": "SEN",
        "iraq": "IRQ",
        "norway": "NOR",
        # Group J
        "argentina": "ARG",
        "algeria": "ALG",
        "austria": "AUT",
        "jordan": "JOR",
        # Group K
        "portugal": "POR",
        "dr congo": "COD",
        "congo dr": "COD",
        "democratic republic of congo": "COD",
        "democratic republic of the congo": "COD",
        "uzbekistan": "UZB",
        "colombia": "COL",
        # Group L
        "england": "ENG",
        "croatia": "CRO",
        "ghana": "GHA",
        "panama": "PAN",
    }

    # Clean up the input string and look up the dictionary
    formatted_country = country.strip().lower()
    return mapping.get(formatted_country, "Code not found")
