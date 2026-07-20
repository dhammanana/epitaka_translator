import sqlite3
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

DB = "/Users/totden/Library/Containers/com.epitaka.epitakaApp/Data/Documents/epitaka.db"              # /Volumes/Data/code/translator/epitaka_translator/data/epitaka.db
OUT = "chunks.parquet"

TARGET = 120
SOFT = 100
LOOKAHEAD = 20
OVERLAP = 40

tokenizer = AutoTokenizer.from_pretrained("dhammanana/harrier-tipitaka-v1")


def n_tokens(text):
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )

print("Loading...")

conn = sqlite3.connect(DB)
df = pd.read_sql("""
SELECT
    book_id,
    para_id,
    line_id,
    vripara,
    pali
FROM sentences
ORDER BY book_id, para_id, line_id
""", conn)
conn.close()

EN_DB = DB.replace("epitaka.db", "epitaka_en.db")
conn = sqlite3.connect(EN_DB)
en_df = pd.read_sql("""
SELECT
    book_id,
    para_id,
    line_id,
    translation
FROM sentences
""", conn)
conn.close()

print("Counting tokens...")

df = df.merge(
    en_df,
    on=["book_id", "para_id", "line_id"],
    how="left"
)

df["translation"] = df["translation"].fillna("")

df["tokens"] = df["pali"].fillna("").apply(n_tokens)

chunks = []

chunk = []
chunk_tokens = 0
previous_overlap = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    row = row.to_dict()

    # first line
    if len(chunk) == 0:
        chunk.append(row)
        chunk_tokens = row["tokens"]
        continue

    # still small
    if chunk_tokens < SOFT:
        chunk.append(row)
        chunk_tokens += row["tokens"]
        continue

    # ------------------------------------
    # Don't start a new verse if we already
    # have a reasonably sized chunk.
    # ------------------------------------
    starts_new_verse = (
        pd.notna(row["vripara"]) and
        len(chunk) > 0
    )

    if chunk_tokens + row["tokens"] <= TARGET:

        # If this row begins a new verse and the current chunk
        # is already large enough, keep the whole verse for
        # the next chunk.
        if starts_new_verse and chunk_tokens >= SOFT:
            pass
        else:
            chunk.append(row)
            chunk_tokens += row["tokens"]
            continue

    # --------------------------
    # choose best cut position
    # --------------------------

    cut = len(chunk)

    # Default: split before the last line
    cut = len(chunk) - 1

    # Try to find a better boundary near the end
    # Prefer cutting at the beginning of the last verse.
    for i in range(len(chunk)-1, 0, -1):

        if pd.notna(chunk[i]["vripara"]):
            cut = i
            break

    # Otherwise cut at paragraph boundary.
    if cut == len(chunk) - 1:
        for i in range(len(chunk)-1, 0, -1):

            back_tokens = sum(x["tokens"] for x in chunk[i:])

            if back_tokens > LOOKAHEAD:
                break

            if chunk[i]["para_id"] != chunk[i-1]["para_id"]:
                cut = i
                break

    # Safety
    cut = max(1, cut)

    save = previous_overlap + chunk[:cut]

    # ---------------------------------------
    # Build overlap for the NEXT chunk
    # ---------------------------------------
    previous_overlap = []

    overlap_tokens = 0

    # Only look at the real chunk, not the previous overlap
    real_save = chunk[:cut]

    for i in range(len(real_save) - 1, -1, -1):

        x = real_save[i]

        # Always include the last line
        if not previous_overlap:
            previous_overlap.insert(0, x)
            overlap_tokens += x["tokens"]
            continue

        # Stop before the beginning of a verse
        if pd.notna(x["vripara"]):
            break

        previous_overlap.insert(0, x)
        overlap_tokens += x["tokens"]

        if overlap_tokens >= OVERLAP:
            break

    chunks.append({
        "book_id": save[0]["book_id"],
        "start_para": save[0]["para_id"],
        "end_para": save[-1]["para_id"],
        "start_line": save[0]["line_id"],
        "end_line": save[-1]["line_id"],
        "token_count": sum(x["tokens"] for x in save),
        "line_count": len(save),
        "pali": "\n".join(x["pali"] for x in save),
        "english": "\n".join(x["translation"] for x in save),
        "text": "\n".join(x["pali"] for x in save)
    })

    chunk = chunk[cut:]
    chunk_tokens = sum(x["tokens"] for x in chunk)

    chunk.append(row)
    chunk_tokens += row["tokens"]

# last chunk
if chunk:
    save = previous_overlap + chunk
    chunks.append({
        "book_id": save[0]["book_id"],
        "start_para": save[0]["para_id"],
        "end_para": save[-1]["para_id"],
        "start_line": save[0]["line_id"],
        "end_line": save[-1]["line_id"],
        "token_count": sum(x["tokens"] for x in save),
        "line_count": len(save),
        "pali": "\n".join(x["pali"] for x in save),
        "english": "\n".join(x["translation"] for x in save),
        "text": "\n".join(x["pali"] for x in save),
    })

chunks = pd.DataFrame(chunks)

chunks_bilingual = chunks.copy()

chunks_bilingual["text"] = (
    "<pali>\n"
    + chunks_bilingual["pali"]
    + "</pali>\n\n<english>\n"
    + chunks_bilingual["english"]
    + "</english>"
)

large = chunks[chunks.token_count > 300]

print(len(large))

print(
    large[
        [
            "book_id",
            "start_para",
            "end_para",
            "token_count",
            "line_count",
        ]
    ].head(50)
)

print(chunks.head())

print()
print("========== Statistics ==========")
print("Chunks:", len(chunks))
print("Min:", chunks.token_count.min())
print("Max:", chunks.token_count.max())
print("Mean:", chunks.token_count.mean())
print("Median:", chunks.token_count.median())

print()
print(chunks.token_count.describe(percentiles=[0.5,0.9,0.95,0.99]))

print()
print(pd.cut(
    chunks.token_count,
    bins=[0,20,40,60,80,100,120,1000]
).value_counts().sort_index())

chunks.to_parquet("chunks.parquet", index=False)
chunks_bilingual.to_parquet("chunks_bilingual.parquet", index=False)

#save to sqlite3
SQLITE_OUT = "chunks.db"
conn = sqlite3.connect(SQLITE_OUT)
chunks.to_sql(
    "chunks",
    conn,
    if_exists="replace",
    index=False,
)
conn.close()
print(f"Saved SQLite: {SQLITE_OUT}")

print(f"\nSaved {OUT}")