"""
Fetch REAL NFT metadata straight from the chain — no API key, no SDK.
=====================================================================

The main tutorial (nft_search.py) ships with a curated sample dataset so it
runs instantly. This script shows how to get the real thing.

It reads `tokenURI(tokenId)` directly off an ERC-721 contract using raw
Ethereum JSON-RPC (just `requests` — no web3.py, no Alchemy/OpenSea key),
resolves the metadata URI (IPFS gateway, plain HTTP, or on-chain `data:`),
normalizes it into the same schema nft_search.py uses, and saves it.

Then it indexes the fetched NFTs into Qdrant and runs a couple of semantic
queries — reusing the exact functions from nft_search.py — to prove the same
code works on real data.

Usage:
    python fetch_nfts.py                          # defaults to Azuki, 12 tokens
    python fetch_nfts.py --contract 0x... --count 20 --start 100
    ETH_RPC_URL=https://your-node python fetch_nfts.py

Public RPCs come and go and rate-limit. The default works at the time of
writing; if you hit errors, pass your own with --rpc or ETH_RPC_URL (e.g.
https://eth.llamarpc.com, or a free Alchemy/Infura node).

Real PFP metadata usually has an empty `description` and rich `attributes`,
so search works on trait combinations: "a character with pink hair holding a
katana" rather than prose. That's the real-world shape of NFT search.
"""

import argparse
import base64
import json
import os
import time
import urllib.parse
from pathlib import Path

import requests

# Function selectors = first 4 bytes of keccak256("signature"). These are
# stable, well-known constants for the ERC-721 metadata interface, so we can
# hardcode them and avoid pulling in a keccak/ABI dependency.
SELECTOR_NAME = "0x06fdde03"      # name()
SELECTOR_TOKEN_URI = "0xc87b56dd"  # tokenURI(uint256)

# Azuki — tokenURI returns an HTTPS metadata URL, so the default run needs no
# IPFS gateway. Override with --contract for any other ERC-721 collection.
DEFAULT_CONTRACT = "0xED5AF388653567Af2F388E6224dC7C4b3241C544"


def rpc_call(rpc_url: str, to: str, data: str) -> str:
    """One read-only `eth_call`. Returns the raw hex result."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    resp = requests.post(rpc_url, json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"RPC error: {body['error']}")
    return body["result"]


def decode_abi_string(hex_result: str) -> str:
    """Decode a dynamic `string` return value from an eth_call result.

    ABI layout: [32 bytes offset][32 bytes length][string bytes]. We don't
    need a full ABI decoder for this one well-defined case.
    """
    raw = bytes.fromhex(hex_result.removeprefix("0x"))
    if len(raw) < 64:
        return ""
    offset = int.from_bytes(raw[:32], "big")
    length = int.from_bytes(raw[offset : offset + 32], "big")
    return raw[offset + 32 : offset + 32 + length].decode("utf-8", errors="replace")


def read_string(rpc_url: str, contract: str, selector: str, token_id: int | None = None) -> str:
    """Call a view function that returns a string (name or tokenURI)."""
    data = selector
    if token_id is not None:
        data += format(token_id, "064x")  # uint256, 32-byte big-endian hex
    return decode_abi_string(rpc_call(rpc_url, contract, data))


def resolve_metadata_uri(uri: str, ipfs_gateway: str) -> dict:
    """Turn a tokenURI into the metadata JSON, whatever scheme it uses."""
    # Fully on-chain NFTs embed the JSON directly as a data: URI.
    if uri.startswith("data:"):
        header, _, payload = uri.partition(",")
        decoded = base64.b64decode(payload).decode() if "base64" in header else urllib.parse.unquote(payload)
        return json.loads(decoded)

    if uri.startswith("ipfs://"):
        path = uri.removeprefix("ipfs://").removeprefix("ipfs/")
        uri = f"{ipfs_gateway.rstrip('/')}/{path}"

    resp = requests.get(uri, timeout=30)
    resp.raise_for_status()
    return resp.json()


def attributes_to_traits(attributes) -> dict:
    """OpenSea-standard `attributes` is a list of {trait_type, value}.
    Flatten it into the {name: value} dict nft_search.py expects."""
    traits: dict = {}
    if isinstance(attributes, list):
        for attr in attributes:
            if not isinstance(attr, dict) or attr.get("value") is None:
                continue
            key = attr.get("trait_type") or attr.get("traitType") or "Trait"
            traits[str(key)] = attr["value"]
    return traits


def fetch_collection(rpc_url, contract, ipfs_gateway, start, count) -> list[dict]:
    contract = contract  # checksum not required for eth_call
    collection_name = read_string(rpc_url, contract, SELECTOR_NAME) or "Unknown Collection"
    print(f"Collection: {collection_name}  ({contract})")

    nfts: list[dict] = []
    for token_id in range(start, start + count):
        try:
            uri = read_string(rpc_url, contract, SELECTOR_TOKEN_URI, token_id)
            meta = resolve_metadata_uri(uri, ipfs_gateway)
        except Exception as exc:  # noqa: BLE001 — keep going on a single bad token
            print(f"  #{token_id}: skipped ({exc})")
            continue

        nfts.append(
            {
                "token_id": token_id,
                "name": meta.get("name") or f"{collection_name} #{token_id}",
                "collection": collection_name,
                "description": meta.get("description", ""),
                "image": meta.get("image", ""),
                "traits": attributes_to_traits(meta.get("attributes", [])),
            }
        )
        print(f"  #{token_id}: {nfts[-1]['name']}  ({len(nfts[-1]['traits'])} traits)")
        time.sleep(0.15)  # be polite to public endpoints

    return nfts


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real NFT metadata from an ERC-721 contract.")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--rpc", default=os.getenv("ETH_RPC_URL", "https://ethereum-rpc.publicnode.com"))
    parser.add_argument("--gateway", default=os.getenv("IPFS_GATEWAY", "https://ipfs.io/ipfs"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", default=str(Path(__file__).parent / "data" / "real_nfts.json"))
    parser.add_argument("--no-search", action="store_true", help="Only fetch; skip the search demo.")
    args = parser.parse_args()

    nfts = fetch_collection(args.rpc, args.contract, args.gateway, args.start, args.count)
    if not nfts:
        raise SystemExit("No NFTs fetched. Try a different --rpc, --gateway, or --contract.")

    Path(args.out).write_text(json.dumps(nfts, indent=2))
    print(f"\nSaved {len(nfts)} NFTs to {args.out}")

    if args.no_search:
        return

    # Reuse the EXACT core from the tutorial — same indexing, same search —
    # to prove the code generalizes from the sample data to real on-chain data.
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient

    from nft_search import MODEL_NAME, build_index, search

    embedder = TextEmbedding(model_name=MODEL_NAME)
    client = QdrantClient(":memory:")
    build_index(client, embedder, nfts)
    print("\nIndexed real NFTs into Qdrant. Try a trait-based semantic query:\n" + "=" * 60)

    for query in ("a character with pink hair", "someone holding a sword or katana", "a spirit or ghost-like figure"):
        print(f'\nQuery: "{query}"')
        for point in search(client, embedder, query, limit=3):
            nft = point.payload
            traits = ", ".join(f"{k}: {v}" for k, v in list(nft["traits"].items())[:4])
            print(f"  {point.score:.3f}  {nft['name']:<16}  {traits}")


if __name__ == "__main__":
    main()
