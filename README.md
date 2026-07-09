# *This project has been created as part of the 42 curriculum by razevedo.*

# CALL ME MAYBE - LLM Function Calling with Constrained Decoding

## Description

This project implements constrained decoding for function calling using a local Large Language Model (LLM) called Qwen/Qwen3-0.6B.

Instead of allowing the model to freely generate arbitrary text, a decoder restricts every generated token so that the output always conforms to a predefined JSON schema describing a valid function call.

The implementation combines:
	
    * A finite-state machine (FSM) representing the JSON grammar
	* Grammar-aware token filtering
	* Incremental token-by-token generation
	* Constrained decoding directly over the model's vocabulary

Rather than validating the output after generation, invalid continuations are eliminated before the model selects its next token.

The objective is **not** to train or fine-tune a language model, but to build the decoding algorithm responsible for producing structured outputs.

---

# Features

* Grammar-constrained decoding
* Finite-state machine representing the JSON structure
* Incremental token validation
* Support for multiple functions schemas
* Dynamic parameter handling
* Compatible with subword tokenization
* Coalescence optimization for deterministic grammar segments
* Automatic generation of valid JSON objects

---

# Project Structure

```bash
src/
    generate.py          # decoding pipeline
    state_machine.py     # FSM states and transitions
    models.py            # Pydantic schemas
    loader.py            # file loading and validation

data/
    input/
    output/

README.md
```

# Instructions

## Installation

```bash
make install
```

## Running

```bash
make run
```

The decoder reads a natural language prompt and generates a structured JSON object describing the selected function and its parameters.

Example:

**Input**

```
What is the sum of 2 and 3?
```

**Output**

```JSON
{
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2.0,
        "b": 3.0
    }
}
```

---

# Algorithm Explanation

The decoding process combines the language model with grammar constraints.

For every generated token:

1. The current state of the finite-state machine determines which parts of the JSON grammar are valid.
2. Those grammar rules are converted into a set of possible string continuations.
3. Because the language model operates on tokens rather than characters, every vocabulary token is checked to determine whether it can still lead to one of those valid strings.
4. Tokens that cannot produce a valid continuation are removed.
5. The language model selects the highest-probability token among the remaining candidates.
6. The finite-state machine advances to its next state.
7. Generation continues until the JSON object is complete.

Since invalid tokens are filtered before selection, malformed JSON cannot be produced by the decoder.

## Handling Tokenization

One of the main challenges is that grammar rules operate on characters while language models generate tokens.

For example, the function name

```
fn_add_numbers
```

may be tokenized as:

```
fn
_add
_numbers
```

or as several other valid token sequences depending on the tokenizer.

The decoder therefore cannot compare complete strings directly.

Instead, it tracks every valid continuation of the grammar and accepts any token whose decoded text remains a valid prefix of at least one possible continuation.

This allows multiple tokenizations of the same string while still enforcing the JSON grammar.

---

## Finite-State Machine

The grammar is represented as a finite-state machine.

Each state corresponds to a specific part of the JSON structure, for example:

```bash

START
   ↓
"name"
   ↓
function name
   ↓
"parameters"
   ↓
parameter key
   ↓
parameter value
   ↓
DONE
```

State transitions are driven by the generated output rather than by fixed token positions, allowing the decoder to support functions with different numbers and types of parameters.

## Coalescence

Many parts of the grammar are deterministic.

For example, once a function name has been completed, the only valid continuation is

```bash
,"parameters":{
```

Instead of querying the language model for every deterministic token, the decoder directly appends these forced segments while keeping the tokenizer state synchronized.

This optimization (known as coalescence) significantly reduces the number of model calls without changing the generated output.

# Design Decisions

Several design choices were made to keep the implementation modular and extensible.

**Finite-state machine**

The JSON grammar is represented independently from the decoding algorithm, making the parser easier to extend with additional parameter types or grammar rules.

**Schema-driven decoding**

Function definitions are loaded from Pydantic schemas rather than hardcoded into the decoder. New functions can therefore be added without modifying the decoding logic.

**Incremental validation**

Instead of validating complete outputs after generation, constraints are applied during decoding so invalid continuations are rejected immediately.

**Tokenizer-aware filtering**

Candidate tokens are validated against possible grammar continuations rather than complete strings, allowing compatibility with subword tokenizers.

---

# Performance Analysis

## Accuracy

The decoder produces function calls that follow the provided schema and correctly extract parameters for the supported function definitions. While grammar constraints guarantee a valid output structure, the semantic correctness of the selected function still depends on the language model's predictions. In practice, the decoder achieves high accuracy on the provided evaluation set while preventing structurally invalid outputs.

## Speed

The first implementation queried the language model for every generated token, leading to execution times between seven and ten minutes.

Introducing coalescence allowed deterministic grammar segments to be generated without additional model calls while preserving compatibility with every valid tokenization.

This reduced execution time to under three minutes while producing identical structured outputs.

## Reliability

Applying grammar constraints during decoding ensures that generated outputs always conform to the implemented JSON structure. Invalid syntax, unknown function names, and malformed parameter objects are rejected before generation can proceed. Additional safeguards detect runaway generation and terminate decoding gracefully if no valid structured output can be produced.

---

# Challenges Faced

The first challenge was the lack of documentation about constrained decoding. Even papers and blogposts that supposedly addressed this technique ommitted the precise part we were supposed to implement. Fortunately, they mentioned a particular library that we were forbidden to use, **outlines**, and fortunately, outlines' authors wrote a paper about their work. So I started there.

Once I understood what constrained decoding was about and after testing how the model worked, the task seemed surprisingly straightforward. The grammar defined exactly which strings were valid at every step, so generating a JSON object appeared to be little more than walking through a finite-state machine.

That assumption quickly broke down once tokenization entered the picture.

The language model does not generate characters or complete words—it generates tokens. A single valid string may correspond to several different token sequences, meaning that matching complete strings was no longer sufficient. The decoder had to reason about prefixes and possible continuations instead, accepting every token that could still lead to a valid output without committing to a particular tokenization.

Another unexpected challenge was performance. My initial implementation queried the language model for every generated token, including deterministic parts of the JSON grammar. While correct, it was unnecessarily slow. Realizing that these forced segments could be appended directly—while still keeping the tokenizer state synchronized—led me to implement coalescence, reducing execution time from several minutes to under three.

Finally, handling parameter values proved more subtle than expected. Numbers have no explicit termination token, strings may legitimately contain punctuation or quotation marks, and different functions accept different parameter types and counts. Keeping the finite-state machine synchronized with partially generated values required several iterations before the decoder behaved reliably across all supported schemas.

---

# Testing Strategy

The decoder was tested using both the provided evaluation suite and additional custom prompts.

Testing covered:

* function selection;
* numeric parameters;
* string parameters;
* multiple parameter types;
* varying parameter counts;
* malformed JSON prevention;
* tokenizer edge cases;
* finite-state transitions;
* empty inputs;
* invalid prompts.

Extensive debugging output was also used during development to inspect state transitions, token filtering, and grammar evolution throughout generation.

---

# Resources

## References

* Brandon T. Willard & Remi Louf, "Efficient Guided Generation for Large Language Models", https://arxiv.org/pdf/2307.09702
* Aidan Cooper, "A Guide to Structured Generation Using Constrained Decoding", https://www.aidancooper.co.uk/constrained-decoding/
* Will Kurt, "Coalescence: making LLM inference 5x faster", https://blog.dottxt.ai/coalescence.html?ref=aidancooper.co.uk#orgf9e4e90
* Celal Savur, "Mastering the Logits: A Guide to Constrained Decoding in Hugging Face and vLLM", https://medium.com/@c.savur/mastering-the-logits-a-guide-to-constrained-decoding-in-hugging-face-and-vllm-357a5c1b9a28
* Let's Data Science Team, "Structured Outputs: Making LLMs Return Reliable JSON", https://letsdatascience.com/blog/structured-outputs-making-llms-return-reliable-json
* Jason Liu, "How to Use Pydantic for LLMs: Schema, Validation & Prompts", https://pydantic.dev/articles/llm-intro
* Pydantic documentation, https://pydantic.dev/docs/validation/dev/api/pydantic/base_model/#pydantic.BaseModel.model_validate_json
* UV documentation, https://docs.astral.sh/uv/
* Python JSON, https://www.w3schools.com/python/python_json.asp
* Martin Fowler, "Function calling using LLMs", https://www.martinfowler.com/articles/function-call-LLM.html

## AI Usage

Artificial intelligence tools were used throughout the project as learning aids rather than code generators.

AI assistance included:

* understanding constrained decoding algorithms;
* discussing finite-state machine design;
* clarifying tokenizer behaviour and subword tokenization;
* explaining theoretical concepts related to grammar-constrained generation;
* reviewing implementation ideas and debugging strategies;
* improving project documentation.

The architecture, implementation, optimization, testing, debugging, and final code were designed and developed by the author.

---

# Future Improvements

Possible extensions include:

* support for recursive JSON schemas;
* beam search with grammar constraints;
* trie-based grammar representation;
* cached grammar transitions;
* faster vocabulary filtering;
* support for nested structured outputs;
* additional JSON schema features.
