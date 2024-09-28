# LLM KG

This repository contains code to construct a knowledge graph using a large language model (LLM), enabling efficient extraction and representation of knowledge from unstructured text.

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone https://gitlab.insight-centre.org/fidelity-project/llm_kg/-/tree/master
   cd llm_kg
   ```

2. **Create and activate a conda environment:**

   ```sh
   conda env create -f environment.yml
   conda activate your-environment-name
   ```

   Ensure that `your-environment-name` matches the name specified in the `environment.yml` file.

## Usage

To run the notebooks and scripts, follow the steps below:

1. **Launch Jupyter Notebook:**

   ```sh
   jupyter notebook
   ```

2. **Open the desired notebook:**

   - `llm_kg.ipynb`: Main notebook demonstrating the setup and usage of the RAG model for Conditional Question Answering.

3. **Execute the cells in the notebook sequentially to extract KG triples and loading Neo4j.**

# Exporting and Importing Neo4j Graphs with APOC

## Prerequisites

- **Neo4j Community Edition** (version 3.2 or higher)
- **APOC Plugin** for Neo4j

## Steps

### 1. Install APOC

1. **Download the APOC `.jar` file** for your Neo4j version.
2. **Place the `.jar` file** in the `$NEO4J_HOME/plugins` directory.
3. **Set permissions**: 
   ```bash
   chmod +rx apoc-3.x.x.x-all.jar
   ```
4. **Update configuration** in `$NEO4J_HOME/conf/neo4j.conf`:
   ```properties
   dbms.security.procedures.unrestricted=apoc.export.*,apoc.import.*
   apoc.export.file.enabled=true
   apoc.import.file.enabled=true
   ```
5. **Restart Neo4j**:
   ```bash
   sudo service neo4j restart
   ```

### 2. Export the Graph

- **Export entire database**:
  ```cypher
  CALL apoc.export.graphml.all("file.graphml", {useOptimizations: true})
  ```

- **Export query results**:
  ```cypher
  CALL apoc.export.graphml.query("MATCH (n) RETURN n", "file.graphml", {useOptimizations: true})
  ```

### 3. Import the Graph

- **Import from file**:
  ```cypher
  CALL apoc.import.graphml("file.graphml", {batchSize: 1000})
  ```

### Notes

- **Node IDs**: During import, original node IDs are not preserved. An `id` property is created instead.

That’s it! For more details, refer to the [APOC documentation](https://neo4j.com/labs/apoc/4.1/).
