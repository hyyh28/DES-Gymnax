以下是整理和修正后的 `README.md` 文档，修复了语言、格式、术语等问题，提升了整体清晰度和专业性：

---

# DES-Gymnax

## 🧩 Introduction

**DES-Gymnax** is a high-performance **Discrete Event Simulator** implemented in [JAX](https://github.com/google/jax). By leveraging JAX's **just-in-time (JIT) compilation**, **automatic vectorization**, and **GPU acceleration**, DES-Gymnax achieves **10x to 100x speedup** compared to traditional Python-based simulators such as *Salabim*.

This simulator provides a **Gym-like API** for seamless integration with **reinforcement learning (RL)** algorithms, bridging the gap between simulation environments and AI/ML applications.

DES-Gymnax has been validated on three benchmark queueing models:

* M/M/1 queue
* Multi-server (M/M/C) queue
* Tandem queue

---

## 📦 Installation

### Requirements

* Python 3.12
* [JAX](https://github.com/google/jax): `pip install jax jaxlib`
* [Matplotlib](https://matplotlib.org/): `pip install matplotlib`
* `tqdm` (optional, for progress bars)

### Setup using Conda

We recommend using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate des-gymnax
```

---

## 🚀 Running the Models

The repository includes implementations of:

* **M/M/1 model**: Run with

  ```bash
  python run_mm1_model.py
  ```

* **M/M/C model**: Run with

  ```bash
  python run_mmc_model.py
  ```

---

## ⚙️ Customizing Simulation Parameters (M/M/1 Example)

You can pass simulation parameters via command-line arguments:

```bash
python run_mm1_model.py \
    --max_time_step 100000 \
    --clerk_processing_time 35 \
    --customers_arriving_time 30
```

### Available Arguments

| Argument                    | Type    | Default | Description                            |
| --------------------------- | ------- | ------- | -------------------------------------- |
| `--max_time_step`           | `int`   | 500000  | Maximum number of time steps           |
| `--clerk_processing_time`   | `float` | 38      | Average time to process a customer     |
| `--customers_arriving_time` | `float` | 40      | Average time between customer arrivals |

---

## 📊 Output

### Console Output

* Average batch rollout execution time
* Average execution time per rollout
* Average customer waiting time
* Average queue length

### Visualization

* **PDF Plot**: A plot of queue dynamics is saved as `MM1_Model.pdf`
* **GIF Export (Optional)**: Uncomment the line in `run_mm1_model.py` to enable GIF generation

---

## 📈 Example Output (4 workers)

```
Average batch rollout execution time: 8.687409 seconds
Average per rollout execution time: 2.171852 seconds
Average waiting time: {0: 397.17021251377844, 1: 368.8520536148124, 2: 378.6416558350486, 3: 413.61272911900556}
```