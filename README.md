# Vector Search for Web3 Developers: Searching NFT Metadata with Qdrant

NFT marketplaces let you filter by exact traits — `Background = Neon City`,
`Rarity = Legendary`. That's great when you know exactly what you want.

But how do you search for *"a brooding warrior glowing with electric light"*
when none of those words appear in the metadata? That's **semantic search**,
and it's what vector databases like [Qdrant](https://qdrant.tech) are built for.

This repo is a complete, runnable example. **No API keys. No Docker. No cloud
account.** Embeddings run locally via [FastEmbed](https://github.com/qdrant/fastembed),
and Qdrant runs in-memory. Clone, install, run.

## Quickstart

```bash
git clone https://github.com/midegdugarova/web3-nft-vector-search.git
cd web3-nft-vector-search
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python nft_search.py
```

The first run downloads a ~130 MB embedding model. After that it's instant.

## What you'll see

```
Query: "a brooding warrior glowing with electric light"
  0.691  Neon Ronin #103        [Common]
  0.670  Neon Ronin #014        [Epic]
  0.643  Neon Ronin #156        [Legendary]

Query: "an adorable soft fluffy companion"
  0.680  Pastel Critter #210    [Rare]
  0.658  Pastel Critter #299    [Common]
  0.655  Pastel Critter #233    [Common]

Query: "a cursed artifact with dark power"
  0.726  Ancient Relic #007     [Legendary]
  0.711  Ancient Relic #019     [Epic]
  0.701  Ancient Relic #044     [Rare]

Query: "powerful and regal" + filter Rarity = Legendary
  0.532  Pastel Critter #251    [Legendary]   <- "wearing a tiny crown"
  0.519  Neon Ronin #156        [Legendary]
  0.501  Neon Ronin #001        [Legendary]
```

None of the query words appear verbatim in the metadata — the model matches on
*meaning*. And the last query shows the killer feature: **semantic search +
trait filtering together**, the marketplace pattern you already know.

## How it works

1. **Load metadata** — `data/nfts.json` holds 15 NFTs across three collections
   (cyberpunk samurai, kawaii animals, mystical relics), each with a name,
   description, and traits. In a real app this comes from IPFS, an indexer like
   The Graph, or a marketplace API.
2. **Embed** — each NFT's text is turned into a 384-dimensional vector that
   captures its meaning, using the local `BAAI/bge-small-en-v1.5` model.
3. **Index** — vectors are stored in Qdrant with the full metadata as payload.
4. **Search** — your query is embedded the same way, and Qdrant returns the
   nearest vectors by cosine similarity.
5. **Filter** — an optional trait filter restricts candidates before ranking,
   combining structured marketplace filtering with semantic relevance.

## Taking it to production

Swap the in-memory client for a real Qdrant instance — that's the only change:

```python
# Local dev (this repo):
client = QdrantClient(":memory:")

# Production — self-hosted via Docker:
#   docker run -p 6333:6333 qdrant/qdrant
client = QdrantClient(url="http://localhost:6333")

# Or Qdrant Cloud:
client = QdrantClient(url="https://YOUR-CLUSTER.qdrant.io", api_key="...")
```

Everything else — indexing, search, filtering — stays identical.

## Fetching real on-chain metadata

The demo ships with curated sample data so it runs instantly. To search a
*real* collection, `fetch_nfts.py` reads `tokenURI` straight off an ERC-721
contract using raw Ethereum JSON-RPC — **no API key, no web3 SDK**, just
`requests`. It resolves IPFS / HTTP / on-chain `data:` URIs, then reuses the
exact `build_index` and `search` functions from `nft_search.py`.

```bash
pip install -r requirements-fetch.txt
python fetch_nfts.py                  # defaults to Azuki, first 12 tokens
# python fetch_nfts.py --contract 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D --count 20
```

```
Collection: Azuki  (0xED5AF388653567Af2F388E6224dC7C4b3241C544)
  #0: Azuki #0  (7 traits)
  ...
Query: "someone holding a sword or katana"
  0.593  Azuki #7    Type: Human, Hair: Orange Samurai, Headgear: Full Bandana...
  0.578  Azuki #10   Type: Human, Hair: Green Samurai, Headgear: Black Bucket Hat...
```

Note the result: querying "katana" surfaced the **Samurai**-haired Azukis —
no shared keyword, pure semantic match. Real PFP metadata usually has an empty
`description` and rich `attributes`, so search runs on trait combinations
("a character with pink hair holding a katana") rather than prose. That's the
real-world shape of NFT search.

> Public RPC endpoints rate-limit and occasionally go down. If the default
> fails, pass your own node with `--rpc` or `ETH_RPC_URL`.

## Web3 use cases beyond this demo

- **Similar-NFT recommendations** — "show me NFTs like this one" by searching
  with an existing token's vector.
- **Natural-language marketplace search** — let users describe what they want.
- **On-chain data search** — embed and search transaction memos, ENS profiles,
  DAO proposals, or governance discussions.
- **Fraud / wash-trading detection** — find anomalous patterns by vector
  distance.

## License

MIT — use it, fork it, ship it.
