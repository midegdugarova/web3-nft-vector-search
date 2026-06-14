"""
Vector Search for Web3 Developers: Searching NFT Metadata with Qdrant
=====================================================================

NFT marketplaces let you filter by exact traits: "Background = Neon City",
"Rarity = Legendary". That works great when you know exactly what you want.

But how do you search for "a brooding warrior glowing with electric light"
when none of those words appear in the metadata? That's *semantic* search,
and it's what vector databases like Qdrant are built for.

This script:
  1. Loads NFT metadata (name, description, traits)
  2. Turns each NFT into an embedding (a vector that captures meaning)
  3. Stores those vectors in Qdrant
  4. Searches by natural language
  5. Combines semantic search WITH trait filtering — the marketplace
     pattern Web3 devs already know, supercharged.

FastEmbed runs the embedding
model locally, and Qdrant runs in-memory. You can install and run without API keys, cloud or docker.
"""

import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# BAAI/bge-small-en-v1.5 is a small, fast, high-quality embedding model.
# It produces 384-dimensional vectors and downloads automatically on first run.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
COLLECTION = "nft_metadata"
DATA_PATH = Path(__file__).parent / "data" / "nfts.json"


def load_nfts() -> list[dict]:
    """Load the NFT metadata. In a real app this would come from
    IPFS, an indexer like The Graph, or a marketplace API."""
    return json.loads(DATA_PATH.read_text())


def nft_to_text(nft: dict) -> str:
    """Embedding models work on text, so we flatten the structured
    metadata into one descriptive string. The description carries the
    'vibe'; the traits add concrete, searchable detail."""
    traits = ", ".join(f"{key}: {value}" for key, value in nft["traits"].items())
    return f"{nft['name']}. {nft['description']} Traits: {traits}."


def build_index(
    client: QdrantClient, embedder: TextEmbedding, nfts: list[dict]
) -> None:
    """Embed every NFT and load it into Qdrant."""
    texts = [nft_to_text(nft) for nft in nfts]
    vectors = list(embedder.embed(texts))  # one 384-dim vector per NFT

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # We store the FULL metadata as the payload. That's what lets us both
    # return rich results AND filter on traits later.
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=nft["token_id"], vector=vector.tolist(), payload=nft)
            for nft, vector in zip(nfts, vectors)
        ],
    )


def search(
    client: QdrantClient,
    embedder: TextEmbedding,
    query: str,
    limit: int = 3,
    query_filter: Filter | None = None,
):
    """Embed the query the same way we embedded the NFTs, then ask Qdrant
    for the closest vectors. An optional filter restricts the candidates
    to NFTs whose traits match — semantic search + marketplace filtering."""
    query_vector = next(embedder.embed([query]))
    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=limit,
    )
    return response.points


def show(title: str, points) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not points:
        print("  (no matches)")
        return
    for point in points:
        nft = point.payload
        print(
            f"  {point.score:.3f}  {nft['name']:<22} [{nft['traits'].get('Rarity', '?')}]"
        )
        print(f"         {nft['description'][:90]}...")


def main() -> None:
    nfts = load_nfts()
    embedder = TextEmbedding(model_name=MODEL_NAME)
    client = QdrantClient(
        ":memory:"
    )  # In production: QdrantClient(url="http://localhost:6333")

    build_index(client, embedder, nfts)
    print(f"Indexed {len(nfts)} NFTs into Qdrant.\n" + "=" * 60)

    # 1. Pure semantic search — note: none of these query words appear
    #    verbatim in the metadata, yet the right NFTs come back.
    show(
        'Query: "a brooding warrior glowing with electric light"',
        search(client, embedder, "a brooding warrior glowing with electric light"),
    )

    show(
        'Query: "an adorable soft fluffy companion"',
        search(client, embedder, "an adorable soft fluffy companion"),
    )

    show(
        'Query: "a cursed artifact with dark power"',
        search(client, embedder, "a cursed artifact with dark power"),
    )

    # 2. Semantic search + trait filter — the marketplace pattern.
    #    "Find me the legendary-tier item that feels powerful and royal."
    legendary_only = Filter(
        must=[FieldCondition(key="traits.Rarity", match=MatchValue(value="Legendary"))]
    )
    show(
        'Query: "powerful and regal" + filter Rarity = Legendary',
        search(client, embedder, "powerful and regal", query_filter=legendary_only),
    )


if __name__ == "__main__":
    main()
