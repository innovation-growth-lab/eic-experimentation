"""
This is a boilerplate pipeline 'generate_synthetic'
generated using Kedro 0.19.1
"""
import re
import ast
import random
import logging
import pandas as pd
import numpy as np
from typing import Dict
from joblib import Parallel, delayed
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

access_token = "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"


def _generate_synthetic_id(input_string):
    """
    Generates a synthetic ID by replacing sequences of digits in the input string
    with random digits.

    Args:
        input_string (str): The input string containing sequences of digits.

    Returns:
        str: The input string with sequences of digits replaced by random digits.

    Example:
        >>> generate_synthetic_id("ABC123XYZ789")
        'ABC456XYZ123'
    """

    digit_sequences = re.findall(r"\d+", input_string)

    for sequence in digit_sequences:
        new_digits = "".join(
            str(random.randint(0, 9)) for _ in range(len(sequence))
        )
        new_sequence = sequence.replace(sequence, new_digits, 1)
        input_string = input_string.replace(sequence, new_sequence, 1)

    return input_string


def _process_item(abstract, paraphrase, template):

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()

    chain = {"sentence": RunnablePassthrough(
    )} | prompt | model | StrOutputParser()
    while True:
        try:
            result = chain.invoke(paraphrase)
            json_result = ast.literal_eval(result)
            return abstract, json_result
        except Exception:
            continue


def _process_address_input(prompt_input, template):

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()

    chain = (
        {"prompt_input": RunnablePassthrough()} | prompt | model | StrOutputParser()
    )

    while True:
        try:
            result = chain.invoke(prompt_input)
            json_result = ast.literal_eval(result)
            name = json_result.get("name", "")
            address = json_result.get("address", "") if len(
                prompt_input) > 1 else ""
            return prompt_input, name, address
        except Exception:
            continue


def _process_keyword_input(prompt_input, template):
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()

    chain = (
        {"prompt_input": RunnablePassthrough()} | prompt | model | StrOutputParser()
    )

    while True:
        try:
            result = chain.invoke(prompt_input)
            json_result = ast.literal_eval(result)
            return json_result
        except Exception:
            continue


def data_processing(data: pd.DataFrame) -> pd.DataFrame:

    id_mapping = {}

    data = data.loc[
        :, ~data.columns.str.contains("^Unnamed")
    ]

    # HACK HACK HACK
    # data = data.sample(10000)

    cols_to_drop = [
        col
        for col in data.columns
        if "CEO" in col and not any(x in col for x in ["ID", "Gender"])
    ] + [col for col in data.columns if "website" in col.lower()]
    data.drop(columns=cols_to_drop, inplace=True)

    # transform dates from five digit to datetime
    for col in data.columns:
        if "date" in col.lower() or "deadline" in col.lower():
            if (
                data[col]
                .apply(
                    lambda x: (isinstance(x, float) or isinstance(x, int))
                    and 10000 <= x < 100000
                )
                .any()
            ):
                data[col] = pd.to_datetime(
                    data[col].apply(
                        lambda x: x if np.isnan(x) else int(x)
                    ),
                    unit="D",
                    origin="1900-01-01",
                )

        # Generate synthetic IDs for the specified columns and store them in id_mapping
        if any(
            keyword in col.lower() for keyword in ["id", "pic", "number"]
        ) and not any(
            keyword in col.lower() for keyword in ["topic", "call", "context"]
        ):
            logger.info(col)
            id_mapping[col] = {}
            unique_ids = data[col].unique()
            for id in unique_ids:
                if id not in id_mapping[col]:
                    id_mapping[col][id] = _generate_synthetic_id(str(id))
            # drop duplicate keys
            for col in id_mapping.keys():
                reversed_dict = {str(v): str(k)
                                 for k, v in id_mapping[col].items()}
                id_mapping[col] = {v: k for k, v in reversed_dict.items()}

            data[col] = data[col].astype(str)

            # replace the IDs in the dataframe with the synthetic IDs
            data[col] = data[col].map(id_mapping[col])

    return data, id_mapping


def synthetic_project_inputs(data: pd.DataFrame) -> pd.DataFrame:
    tokeniser = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def _process_abstract(abstract):
        anonymised_abstract = re.sub(r"\b[A-Z]\w*\b", "ENTITY", str(abstract))
        inputs = tokeniser.encode(
            "summarize: " + anonymised_abstract, return_tensors="pt")
        first_outputs = model.generate(
            inputs, max_length=150, num_return_sequences=1, temperature=1)
        paraphrase = tokeniser.decode(first_outputs[0], skip_special_tokens=True)
        return abstract, paraphrase

    logger.info(
        "Abstract Input Creation - Initialising",
    )

    # get all unique project abstract, date pairs
    abstract_column = next(
        (col for col in data.columns if "abstract" in col.lower()), None
    )

    unique_projects = data[[abstract_column]].drop_duplicates()

    outputs = Parallel(n_jobs=10, verbose=10)(
        delayed(_process_abstract)(abstract) for abstract in unique_projects[abstract_column]
    )

    return outputs


def get_synthetic_projects(outputs: list) -> Dict:
    """
    Generate synthetic projects based on a given prompt using cutting-edge technology.
    The generated projects will follow a specific structure and include fields such as
    Acronym, Title, and Abstract.
    """
    template = """

    Generate a JSON-like object describing a cutting-edge tech project in a similar 
    style to the examples below from a short prompt. The examples are given in JSON format, and the output 
    should also follow the same structure. The generated project should embody innovation, 
    ambition, and professionalism. Use clear and concise language, provide specific 
    details about the technology, partnerships, and market potential, and highlight 
    the project's unique impact, funding status, and strategic partnerships. Please remove
    any references to a specific company or product name. Make sure the output is heavily
    rewritten, so that the abstract is not fuzzy matchable to the input. 

    Examples:
    
    - input: "ENTITY is an AI-based autonomous flight control for the aircraft of today and electric VTOLs"
    - output: {{
        "Acronym": "Machbel",
        "Title": "AI-based autonomous flight control for the aircraft of today and electric VTOLs of tomorrow",
        "Abstract": "We are developing Machbel, a novel AI autonomous flight control system. Machbel is an airframe-agnostic solution targeted at fly-by-wire aircraft of today (fixed-wing airplanes & helicopters) & tomorrow (eVTOLs & others). It is a combination of custom-designed neural networks, core avionics software, computer vision algorithms & special-purpose aviation-grade hardware. We are developing a special kind of neural networks that are deterministic by design, built specifically for the purposes of certification in aviation. They learn while they are getting trained, they don’t learn when they are piloting aircraft. They run in the dedicated environment & on certifiable hardware that we are creating. This is a major innovation over the generic state-of-the-art in AI.Airbus forecasts that the potential demand for eVTOLs in the urban air mobility market will be 100X bigger its current yearly production of helicopters. This market can only grow with full flight autonomy provided by Raven.We have raised €4m+ of funding & already generated €110,000 in early revenues. Our current deal pipeline is worth €2m. As of May 2019 we have signed agreements with 14 most prominent eVTOL OEMs (out of 85+) & have been approached by dozens of others. We are already working with high-profile eVTOL manufacturers such as Volocopter, Lilium Aviation, Kitty Hawk, Dufour Aerospace & multiple others. We also have closed commercial agreements with the avionics vendors, drone OEMs & traditional aircraft OEMs such Euroavionics, AirTractor & Airbus Helicopters. We have also entered a strategic commercial partnership with a major German car manufacturer that wants to enter the eVTOL market & sees Raven as the key enabling solution.Starting with 2020 we forecast steady cashflows with the drones & existing general aviation market segments. To reach a unicorn valuation we are betting on the urban air mobility market segment."
    }}

    - input: "ENTITY is a company committed to reducing the Textile Industry defective production (from 5%) to 0%"
    - output: {{
        "Acronym": "PUTLA",
        "Title": "Detection of defective textile production",
        "Abstract": "PUTLA seeks to reduce waste in textiles, an endemic issue affecting some 5% of the production. Our dedicated and highly skilled team has made this possible by developing unique devices, which combine hardware and software built in-house to achieve such a goal. SMARTEX is able to detect defects during the production stage, offering an online software as a service (SaaS) based on Computer Vision, Machine Learning and Artificial Intelligence with the clear mission of becoming the main solution for automation of the inspection processes in the Knitting Textile Industry. Currently, SMARTEX devices are in a prototype stage (TRL 6), since real short-time environment tests have been successfully performed and the market-fit is also validated by multiple LOIs and prices agreed with potential clients, including Decathlon group, PVH (Calvin Klein, Tommy Hilfiger, etc.), Kering (Gucci, YSL), in different continents. SMARTEX’s disruptive hardware & software technology has been recognised with several grants and awards and has recently participated in the biggest hardware startup acceleration programme HAX (Shenzhen, China), backed by Smartex’s first investor SOSV (Sean O’Sullivan Ventures)."
    }}

    - input: "Entity is an innovative autonomous solar drone dedicated to earth observation."
    - output: {{
        "Acronym": "Massimmo",
        "Title": "Earth Observation by Autonomous Solar UAV",
        "Abstract": "MAsla develops an autonomous solar drone that can help improve monitoring of the planet. Inspired from satellite earthobservation, MAsla aims to offer affordable earth data acquisition performed by unmmaned autonomous vehicules to arange of end users which have all express their support to the SolarXOne project : Linear infrastructure observation (such asrailway, pipeline or electrical grid), environmental & security surveillance issues (forest fire detection, traffic surveillance),Maritime observation (traffic surveillance, fishing surveillance), precision agriculture (monitoring of the health of crops andlivestock). A first autonomous prototype is ready (TRL6). Thanks to a patented double-wing innovative design, theperformances of SolarXOne are disruptive compared to existing solutions: large payload capacity (7kg) enabling to carry awide range of data acquisition sensors, very stable flight enabling precise data acquisition, long flight (>600 km / Day),cheaper price for end user compared to competitors. A world record of autonomous solar flight will be tempted in the nextmonths. SolarXOne project objectives are to industrialise the drone production with enhance performances. Adaptability toHydrogen energy source will be added as well as vertical take-off capability. Market demonstration will be done during theproject for linear infrastructure, fire detection, maritime surveillance and precision agriculture. Two operating centres will beopened in France and Germany. XSun is based on 2 complementary business models: earth data service commercialisationoperated from the control centres and complete system commercialisation. Market analysis have been performed for eachsegment and the business plan shows promising revenue reaching 30 M€ in 2025. MAsla has been created in 2016, its team (12 people) is composed of experienced managers and business developers in the aerospace industry as well as young passionate engineers."
    }}

    Make sure your generated project follows the same tone and structure, as JSON files. Include these fields:
    - Acronym
    - Title
    - Abstract

    Go ahead, produce the output for the following example:
    - input: {sentence}
    - output:
    """

    results = Parallel(n_jobs=8, verbose=10)(
        delayed(_process_item)(abstract, paraphrase, template) for abstract, paraphrase in outputs
    )

    output = {abstract: json_result for abstract, json_result in results}

    return output


def organisation_randomiser(data: pd.DataFrame) -> Dict:
    """
    Randomises the names and addresses of organisations based on existing ones in
    the dataframe.

    This method generates random names and addresses based only vaguely on existing
    ones in the dataframe. It takes into account the columns containing the legal
    names and addresses of organisations. If both name and address columns are present,
    it generates random names and addresses together. If only the name column is
    present, it generates random names only.

    Examples:
    - input: ["DAEDALEAN AG", "Wattstr. 11"]
        output: {"name": "Schwarz GK", "address": "WILHELMINA VAN PRUISENWEG 35"}

    - input: ["FEOPS NV", "WILHELMINA VAN PRUISENWEG 35"]
        output: {"name": "Klein AG", "address": "Frederik Strasse 1"}

    """
    name_cols = [
        col for col in data.columns if "legal" in col.lower() and "name" in col.lower()
    ]
    address_cols = [
        col for col in data.columns if "address" in col.lower()
    ]

    if not name_cols and not address_cols:
        return {}
    if name_cols and address_cols:
        unique_inputs = (
            data[name_cols + address_cols]
            .drop_duplicates()
            .values.tolist()
        )

        template = """

        Generate random names and addresses based only vaguely on existing ones.

        Examples:

        - input: ["DAEDALEAN AG", "Wattstr. 11"]
        - output: {{"name":"Schwarz GK", "address": "WILHELMINA VAN PRUISENWEG 35"}}

        - input: ["FEOPS NV", "WILHELMINA VAN PRUISENWEG 35"]
        - output: {{"name":"Klein AG", "address": "Frederik Strasse 1"}}

        Go ahead, produce the output for the following example:
        - input: {prompt_input}
        - output:
        """
    else:
        unique_inputs = data[name_cols].drop_duplicates().values.tolist()

        template = """

        Generate random names and addresses based only vaguely on existing ones.
    

        Examples:
        
        - input: "DAEDALEAN AG"
        - output: {{"name":"Schwarz GK"}}

        - input: "FEOPS NV"
        - output: {{"name":"Klein AG"}}

        Go ahead, produce the output for the following example:
        - input: {prompt_input}
        - output:
        """

    results = Parallel(n_jobs=8)(
        delayed(_process_address_input)(prompt_input, template) for prompt_input in unique_inputs
    )

    return results


def keyword_randomiser(data: pd.DataFrame) -> Dict:
    """
    Randomises keywords in a DataFrame based on a set of input keywords.

    Args:
        data (pd.DataFrame): The DataFrame containing the keywords to be randomised.

    Returns:
        pd.DataFrame: The DataFrame with the randomised keywords.

    """
    keyword_cols = [
        col for col in data.columns if "keyword" in col.lower() or " kt " in col.lower()
    ]

    if not keyword_cols:
        return {}

    unique_inputs = data[keyword_cols].drop_duplicates().values.tolist()

    template = """

    Generate random keywords based only vaguely on a set of input ones.

    Examples:

    - input: ["Energy", "Renewable energy sources", "Engineering and technology", "Control engineering"]
    - output: {{"Energy":"Electricity", "Renewable energy sources":"Solar power", "Engineering and technology":"Aerospace engineering", "Control engineering":"Control systems"}}

    - input: ["Autonomous flight control, urban air mobility, drones, eVTOL, computer vision", "Artificial intelligence"]
    - output: {{"Autonomous flight control, urban air mobility, drones, eVTOL, computer vision":"Kite, aviation, aerospace, flight control, autonomous flight", "Artificial intelligence":"Machine learning"}}
    
    Go ahead, produce the output for the following example:
    - input: {prompt_input}
    - output:
    """

    results = Parallel(n_jobs=8)(
        delayed(_process_keyword_input)(prompt_input, template) for prompt_input in unique_inputs
    )
    combined_results = {}
    for result in results:
        combined_results.update(result)

    return combined_results


def zip_randomiser(data: pd.DataFrame):
    """
    Randomises the digits in a column containing zipcodes.

    This method identifies a column in the dataframe that includes a zipcode and
    randomises the digits in that column. It searches for a column with the word
    "zipcode" (case-insensitive) in its name. If such a column is found, it applies
    the `randomise_zipcode` function to each value in that column.

    The `randomise_zipcode` function randomly selects a digit in the zipcode and
    replaces it with a random digit from 0 to 9. The modified zipcode is then
    assigned back to the dataframe.

    After randomising the zipcodes, the method proceeds to the `shuffle_amounts`
    step.
    """
    # identify a possible column that includes a zipcode
    zipcode_col = next(
        (col for col in data.columns if "zipcode" in col.lower()), None
    )

    if zipcode_col:
        # identify any numerics and change one random number
        def randomise_zipcode(zipcode):
            zipcode = str(zipcode)
            if re.search(r"\d", zipcode):
                random_index = random.choice(
                    [m.start() for m in re.finditer(r"\d", zipcode)]
                )
                new_digit = str(random.randint(0, 9))
                return (
                    zipcode[:random_index] + new_digit +
                    zipcode[random_index + 1:]
                )
            return zipcode

        data[zipcode_col] = data[zipcode_col].apply(
            randomise_zipcode
        )

    return data


def shuffle_status(data: pd.DataFrame):
    """
    Shuffles the values in columns containing the word "status" in their name.

    This method shuffles the unique values in each column that contains the word "status"
    in its name. It uses numpy's random permutation function to generate a new order for
    the unique values,and then maps the old values to the new shuffled values in the
    dataframe.

    After shuffling the status columns, the method proceeds to the next step, which is
    saving the dataframe to S3.

    """
    status_cols = [col for col in data.columns if "status" in col.lower()]

    # block reassignment
    for col in status_cols:
        unique_values = data[col].unique()
        shuffled_values = np.random.permutation(unique_values)
        mapping_dict = dict(zip(unique_values, shuffled_values))

        # Map the old values to the new shuffled values
        data[col] = data[col].map(mapping_dict)

    return data


def synthetise_data(data: pd.DataFrame, content_synthetics: Dict, orgs_synthetics: Dict, keywords_synthetics: Dict) -> pd.DataFrame:
    """
    Synthetises data based on provided content, organisations, and keywords synthetics.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be synthetised.
        content_synthetics (dict): A dictionary mapping original content to synthetised content.
        orgs_synthetics (dict): A dictionary mapping prompt inputs to synthetised organisation names and addresses.
        keywords_synthetics (dict): A dictionary mapping original keywords to synthetised keywords.

    Returns:
        pd.DataFrame: The synthetised DataFrame.

    """
    abstract_column = next(
        (col for col in data.columns if "abstract" in col.lower()), None
    )
    title_column = next(
        (col for col in data.columns if "title" in col.lower()), None
    )
    acronym_column = next(
        (col for col in data.columns if "acronym" in col.lower()), None
    )

    # iterate over the DataFrame rows
    for i, row in data.iterrows():
        abstract = row[abstract_column]

        # match the abstract to the specific rows in output
        if abstract in content_synthetics:
            result = content_synthetics[abstract]
            for key, col in [
                ["Acronym", acronym_column],
                ["Title", title_column],
                ["Abstract", abstract_column],
            ]:
                try:
                    # replace the values in the acronym, title, and abstract columns
                    data.at[i, col] = result.get(
                        key, data.at[i, col]
                    )
                except:
                    pass

    # do the same with orgs
    name_cols = [
        col for col in data.columns if "legal" in col.lower() and "name" in col.lower()
    ]
    address_cols = [
        col for col in data.columns if "address" in col.lower()
    ]

    output_names = {}
    output_addresses = {}

    for prompt_input, name, address in orgs_synthetics:
        output_names[prompt_input[0]] = name
        if address:
            output_addresses[prompt_input[1]] = address

    data.reset_index(drop=True, inplace=True)

    # make sure output_names has unique keys
    reversed_dict = {v: k for k, v in output_names.items()}
    output_names = {v: k for k, v in reversed_dict.items()}

    for col in name_cols:
        data[col] = data[col].map(output_names)
    if address_cols:
        for col in address_cols:
            data[col] = data[col].apply(lambda x: output_addresses.get(x, ""))

    # and keywords
    keyword_cols = [
        col for col in data.columns if "keyword" in col.lower() or " kt " in col.lower()
    ]
    for col in keyword_cols:
        # Apply the mapping to each cell in the column
        data[col] = data[col].apply(lambda x: keywords_synthetics.get(x, x))

    return data
