# Prompts of the 2026-11-19 activation-precision run (ds4n-glm53-oracle-run --activations fp32,bf16,e4m3,mxfp4)
# Each is [gMASK]<sop> (154822,154824) + the text, encoded with kitaniai's tokenizer.json, add_special_tokens=False.

## prompt 0 (21 tokens): the capitals prompt of the first run
The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is

## prompt 1
The water cycle describes how water evaporates from oceans and lakes, rises into the atmosphere, cools and condenses into clouds, and falls back to the surface as rain or snow. Over land, some of that water soaks into the soil and recharges groundwater, while the rest flows through rivers back to the sea.

## prompt 2
def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

## prompt 3
Question: A train travels 180 kilometres in 2 hours, then 120 kilometres in 1.5 hours. What is its average speed for the whole journey?
Answer: The total distance is 180 + 120 = 300 kilometres. The total time is 2 + 1.5 = 3.5 hours. So the average speed is 300 / 3.5, which is about 85.7 kilometres per hour.

## prompt 4
长城是中国古代的军事防御工程，修建历史可以追溯到春秋战国时期。明朝时期对长城进行了大规模的修缮和扩建，今天我们看到的大部分长城都是明长城。
