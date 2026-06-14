---
title: "Vector Search for Web3 Developers: Searching NFT Metadata with Qdrant"
published: false
tags: web3, ai, nft, python
canonical_url:
---

If you've built anything on-chain, you know how NFT search works today: exact-match
filters. `Background = Neon City`. `Rarity = Legendary`. `Eyes = Laser`. Marketplaces
are basically faceted databases — pick your traits, get your grid.

That's perfect when you know exactly what you want. But it falls apart the moment a
user thinks in *vibes* instead of attributes:

> "Show me a brooding warrior glowing with electric light."

There's no `vibe = brooding` trait. The words "brooding," "glowing," and "electric"
might not appear in a single NFT's metadata. Exact-match search returns nothing.

This is the gap **vector search** fills — and it's a tool most Web3 developers
haven't reached for yet. I'm going to show you how to add semantic search to NFT
metadata in about 40 lines of Python, with **no API keys, no Docker, and no cloud
account**. Then I'll show you the part that actually matters for marketplaces:
combining semantic search with the trait filters you already use.

> **Who I am, briefly:** I spent the last few years doing developer relations in
> blockchain. I'm now working in AI infrastructure, and the overlap between the two
> worlds is bigger than either side realizes. This post is one example.

## The idea in one sentence

Turn each NFT's text into a list of numbers (an *embedding*) that captures its
meaning, store those numbers in a vector database, and search by *meaning* instead
of by exact string match.

If you've heard "embeddings" and "vectors" thrown around and tuned out — that's the
whole concept. A model reads "a fluffy lavender bunny in cotton-candy clouds" and
produces a 384-number fingerprint. Two NFTs with similar meaning get similar
fingerprints, even if they share no words. Search becomes "find the closest
fingerprints."

## The stack (and why it's zero-friction)

- **[Qdrant](https://qdrant.tech)** — an open-source vector database written in Rust.
  We'll run it in-memory so there's nothing to install or host.
- **[FastEmbed](https://github.com/qdrant/fastembed)** — runs the embedding model
  locally. No OpenAI key, no rate limits, no per-call cost.

That combination matters. Every "intro to vector search" tutorial I tried as a
newcomer wanted an OpenAI key, a Pinecone account, *and* a Docker daemon before I
could see a single result. Here you clone and run.

```bash
pip install "qdrant-client[fastembed]"
```

## Step 1: The data

Real NFT metadata lives on IPFS or comes from an indexer like The Graph. For the
demo, `data/nfts.json` has 15 NFTs across three collections — cyberpunk samurai,
kawaii animals, and mystical relics — each shaped like standard marketplace metadata:

```json
{
  "token_id": 1,
  "name": "Neon Ronin #001",
  "collection": "Neon Ronin",
  "description": "A masterless samurai cloaked in a rain-soaked trench coat, his katana humming with electric blue plasma...",
  "traits": {"Background": "Neon City", "Weapon": "Plasma Katana", "Armor": "Trench Coat", "Rarity": "Legendary"}
}
```

## Step 2: Turn metadata into something searchable

Embedding models read text, so we flatten the structured metadata into one string —
the description carries the vibe, the traits add concrete detail:

```python
def nft_to_text(nft: dict) -> str:
    traits = ", ".join(f"{k}: {v}" for k, v in nft["traits"].items())
    return f"{nft['name']}. {nft['description']} Traits: {traits}."
```

## Step 3: Embed and index

```python
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # 384-dim, local
client = QdrantClient(":memory:")                              # nothing to host

client.create_collection(
    collection_name="nft_metadata",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

texts = [nft_to_text(n) for n in nfts]
vectors = list(embedder.embed(texts))

client.upsert(
    collection_name="nft_metadata",
    points=[
        PointStruct(id=n["token_id"], vector=v.tolist(), payload=n)
        for n, v in zip(nfts, vectors)
    ],
)
```

We store the **full metadata** as the payload. That's what lets us return rich
results *and* filter on traits in a moment.

## Step 4: Search by meaning

```python
def search(query, limit=3, query_filter=None):
    qv = next(embedder.embed([query]))
    return client.query_points(
        collection_name="nft_metadata",
        query=qv.tolist(),
        query_filter=query_filter,
        limit=limit,
    ).points
```

Now the payoff. Remember: **none of these query words appear verbatim in the
metadata.**

```
Query: "a brooding warrior glowing with electric light"
  0.691  Neon Ronin #103   A wandering swordsman bathed in soft teal light...
  0.670  Neon Ronin #014   A cybernetic warrior with a chrome jaw and glowing red optics...
  0.643  Neon Ronin #156   An armored general clad in glowing crimson nano-plates...

Query: "an adorable soft fluffy companion"
  0.680  Pastel Critter #210   An impossibly fluffy lavender bunny...
  0.658  Pastel Critter #299   A sleepy yellow duckling curled inside a teacup...

Query: "a cursed artifact with dark power"
  0.726  Ancient Relic #007   A weathered golden amulet inscribed with forgotten runes...
  0.711  Ancient Relic #019   A cracked obsidian dagger... humming with dark energy.
```

Three vibe-based queries, three clean separations across collections. The model
understood "brooding warrior" maps to samurai, "fluffy companion" maps to cute
animals, and "cursed artifact" maps to the obsidian necrotic dagger — without a
single shared keyword.

## Step 5: The part that matters for marketplaces

Pure semantic search is a nice demo. But marketplaces live on trait filters, and
your users won't give those up. The good news: **you don't have to choose.** Qdrant
filters the candidate set by traits *and* ranks by semantic similarity in one query.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

legendary_only = Filter(
    must=[FieldCondition(key="traits.Rarity", match=MatchValue(value="Legendary"))]
)

search("powerful and regal", query_filter=legendary_only)
```

```
Query: "powerful and regal" + filter Rarity = Legendary
  0.532  Pastel Critter #251   A chubby peach-colored hamster wearing a tiny crown...
  0.519  Neon Ronin #156       An armored general clad in glowing crimson nano-plates...
  0.501  Neon Ronin #001       A masterless samurai... katana humming with electric blue plasma.
```

Two things happened here. First, the filter did its job — *only* Legendary-tier NFTs
came back. Second, and this is my favorite result: the **top hit is a crowned
hamster**. The model connected "regal" to "wearing a tiny crown" — across the
cute/fierce divide, with zero shared words. That's the difference between matching
strings and matching meaning.

This is the mental model shift for Web3 devs: your existing trait filters become the
*structured* layer, and vector search adds a *semantic* layer on top. Same query, both
worlds.

## Going to production

The only line that changes is the client:

```python
# Local dev:
client = QdrantClient(":memory:")

# Self-hosted:  docker run -p 6333:6333 qdrant/qdrant
client = QdrantClient(url="http://localhost:6333")

# Qdrant Cloud:
client = QdrantClient(url="https://YOUR-CLUSTER.qdrant.io", api_key="...")
```

Indexing, search, and filtering are identical.

## Bonus: pointing it at a real collection

The sample data is curated so the demo runs instantly, but you'll want real
metadata. Here's the part I like as a Web3 dev: you don't need OpenSea's API, an
Alchemy key, or even web3.py. NFT metadata lives on-chain — just read `tokenURI`
off the contract with a plain JSON-RPC call.

```python
import requests

SELECTOR_TOKEN_URI = "0xc87b56dd"  # keccak256("tokenURI(uint256)")[:4]

def token_uri(rpc_url, contract, token_id):
    data = SELECTOR_TOKEN_URI + format(token_id, "064x")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
               "params": [{"to": contract, "data": data}, "latest"]}
    result = requests.post(rpc_url, json=payload, timeout=30).json()["result"]
    # decode the ABI string: [32b offset][32b length][bytes]
    raw = bytes.fromhex(result[2:])
    length = int.from_bytes(raw[32:64], "big")
    return raw[64:64 + length].decode()
```

Resolve the URI (it'll be `ipfs://`, an HTTPS gateway, or an on-chain `data:`
URI), fetch the JSON, flatten its `attributes`, and index it exactly like before.
The repo's `fetch_nfts.py` does all of this and then runs the same search on real
[Azuki](https://www.azuki.com/) tokens:

```
Query: "someone holding a sword or katana"
  0.593  Azuki #7    Hair: Orange Samurai, Headgear: Full Bandana...
  0.578  Azuki #10   Hair: Green Samurai, Headgear: Black Bucket Hat...
```

The query said "katana"; the results are the **Samurai**-haired Azukis. No shared
word — the model just understood the connection. One honest caveat worth knowing:
real PFP collections usually leave `description` empty and put everything in
`attributes`, so semantic search runs over trait *combinations* ("a character with
pink hair holding a katana") rather than prose. That's the real shape of NFT
metadata, and vector search handles it cleanly.

## Where this goes next

NFT metadata is the friendly on-ramp, but the same pattern unlocks a lot of Web3
problems that exact-match search can't touch:

- **"NFTs like this one"** recommendations — search with an existing token's vector.
- **Natural-language marketplace search** — let users describe what they want.
- **On-chain text search** — ENS profiles, DAO proposals, governance threads.
- **Wash-trading / anomaly detection** — find outliers by vector distance.

The full, runnable code is on GitHub: **[link your repo here]**. Clone it, point it
at a real collection's metadata, and you've got semantic NFT search in an afternoon.

If you're building at the Web3 × AI intersection, I'd genuinely like to hear what
you're working on — find me at [your handle].
