# EIC Datathon

The EIC has partnered with IGL to harness data and experimental approaches for enhancing policy design and funding mechanisms. This collaboration aims to optimise the use of internal and external data, identifying trends, and promoting research and innovation more effectively.

### Why a Datathon
- **Exploring rich data**: The datathon offers researchers the chance to dive into and analyse data from one of the EU’s largest innovation funders. It is a chance to tackle relevant and pressing questions in the field of innovation technology, leveraging the wealth of EIC data.

- **Shaping future access**: The EIC is eager to make its data available for impactful research, but this requires navigating the complexities of permissions and data use. This datathon serves as a sandbox for exploring how to share data responsibly and effectively.

- **Driving impact**: This is an opportunity to influence how innovation data is utilised at a policy level. Insights can guide future data sharing practices and researcher collaborations. 

### Why Asynchronous
- **Confidentiality**: Due to the sensitive nature of the data, direct access cannot currently be granted. Instead, IGL will set up of a virtual Secure Research System (vSRS), allowing researchers to submit code tested on synthetic data. This compromise enables work with the data while adhering to existing limitations, with the aim of expanding access in the future.

- **Time commitment**: The format is designed to mimic the engagement level of a short datathon event, but spread over a longer period. This should allow participants to thoroughly explore and develop their ideas alongside other existing commitments. 

## Generator Metaflow
This section explains how to use the `SyntheticDataGenerator` class to generate synthetic data from annotated proposals and taxonomy data.

### Overview
The `SyntheticDataGenerator` is a Metaflow flow specification designed to generate synthetic data using pre-trained models. It does this in several steps:

1. **Load annotated proposals and taxonomy data**.
2. **Process the data**, transform dates, and generate synthetic IDs.
3. **Generate synthetic project inputs** by paraphrasing project abstracts.
4. **Generate JSON-like objects** that describe cutting-edge tech projects.
5. **Randomise organisation names and addresses**.
6. **Shuffle sensitive information** like amounts and statuses.

### Usage Instructions

**Run the flow using the `run` method**:

```bash
python -m eic_datathon.pipeline.synthetic_generation.generator_metaflow run --environment pypi --files_to_load datathon/real/acc_h20_proposals.xlsx,datathon/real/acc_he_projects.xlsx --max-workers 2
```

### Data Changes Summary
The data is shuffled and rewritten but remains representative of the original distribution:

- **Shuffling**: Sensitive columns are shuffled rather than removed, preserving location (approximate), topic, decision, and grant amount distributions.
- **Rewriting**: Project proposals are paraphrased instead of replaced with random text. This is done by paraphrasing using local models (T5 instance) the first 50 characters, and then using OpenAI to populate the paraphrased sentence into full-size abstracts.
- **Anonymising**: Individual information (PIC, BvD, company name, address, zipcode, etc.) is fully anonymised.

### Flow Steps
#### 1. `start`: Load Annotated Proposals and Taxonomy Data
- **Files Loading**: Loads annotated proposals and taxonomy data.
- **Seed Setting**: Fixes seeds for consistent synthetic data generation.
- **Data Loading**:
  - Loads all `.xlsx` files if no specific files are provided.
  - Otherwise, loads the specified files.

#### 2. `data_processing`: Process Loaded Data
- **Drop Unnamed Columns**: Drops any columns with "Unnamed" in their name.
- **Remove Personal Info**: Deletes columns containing "CEO" except for those with "ID" or "Gender."
- **Date Transformation**: Converts five-digit dates to datetime format.
- **Generate Synthetic IDs**:
  - Generates synthetic IDs for columns with "ID," "PIC," or "Number."
  - Maps original IDs to new synthetic IDs.

#### 3. `synthetic_project_inputs`: Generate Synthetic Project Inputs
- **Paraphrase Project Abstracts**:
  - Uses the T5 model to generate synthetic abstracts.
  - Takes each unique project abstract and generates a paraphrased version using the T5 model.

#### 4. `get_synthetic_projects`: Generate JSON-Like Objects
- **Create JSON Objects**:
  - Uses a language model to generate JSON-like objects for tech projects based on paraphrased inputs.
  - Outputs include "Acronym," "Title," and "Abstract."

#### 5. `organisation_randomiser`: Randomise Organisation Names and Addresses
- **Random Name/Address Generation**:
  - Generates random organisation names and addresses based on existing ones.

#### 6. `zip_randomiser`: Randomise Zipcodes
- **Randomise Digits in Zipcodes**:
  - Identifies a column containing zip codes and randomises their digits.

#### 7. `shuffle_amounts`: Shuffle Amount-Related Columns
- **Shuffle Amount Values**:
  - Shuffles values in columns like "amount," "grant," or "costs."

#### 8. `shuffle_status`: Shuffle Status Columns
- **Shuffle Status Values**:
  - Shuffles unique values in status columns.

#### 9. `save_to_s3`: Save to S3
- **Save Dataframe**:
  - Saves the processed data and ID mapping to S3.

#### 10. `join`: Collect Results
- **Join Inputs**:
  - Collects inputs from previous steps.

#### 11. `end`: End of Flow
- Marks the end of the flow.

### Example
```python
generator = SyntheticDataGenerator(files_to_load="data.xlsx")
generator.run()
```

