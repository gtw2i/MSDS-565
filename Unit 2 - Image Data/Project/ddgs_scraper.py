"""
DuckDuckGo image scraper for MSDS 565 P2.

Build a labeled image dataset by searching DuckDuckGo Images, downloading the
results, filtering out images that are too small or the wrong shape, and saving
what survives into one folder per class.

    from ddgs_scraper import build_image_dataset

    records, summary = build_image_dataset(
        class_queries={
            "hammer": ["claw hammer", "ball peen hammer", "sledgehammer"],
            "wrench": ["adjustable wrench", "socket wrench", "pipe wrench"],
        },
        save_dir="images",
        images_per_class=250,
    )

`records` is one row per saved image and is what you turn into the manifest CSV.

Requires the `ddgs` package:  pip install ddgs
"""

import random
import re
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from requests.exceptions import ConnectionError, RequestException, Timeout

from ddgs import DDGS


# ================================================================
# 1. Validating class names and queries
# ================================================================
# A class label is used in two places on disk:
#   1. As a sub-directory name:  save_dir/<label>/
#   2. As a filename prefix:     save_dir/<label>/<label>_000.jpg
# The same character rules apply to both, so one validator covers both.

# Characters that are illegal in folder/file names on Windows or Unix
_ILLEGAL_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

# Reserved device names on Windows (case-insensitive)
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Conservative cross-platform limit, well under 255 so there is room for the
# "_000.jpg" suffix (8 characters) that gets appended to filenames.
_MAX_NAME_LEN = 200


def validate_label_as_folder_name(label: str) -> str:
    """
    Check that a class label can be used as both a folder name and a filename
    prefix. Returns the stripped label on success; raises ValueError otherwise.
    """
    if not isinstance(label, str):
        raise TypeError(f"Label must be a str, got {type(label).__name__!r}: {label!r}")

    stripped = label.strip()

    if not stripped:
        raise ValueError(f"Label is empty or whitespace-only: {label!r}")

    illegal = re.findall(_ILLEGAL_CHARS, stripped)
    if illegal:
        chars = ", ".join(repr(c) for c in sorted(set(illegal)))
        raise ValueError(
            f"Label {stripped!r} contains characters that are illegal in folder "
            f"and file names: {chars}"
        )

    if stripped.upper() in _WINDOWS_RESERVED:
        raise ValueError(
            f"Label {stripped!r} is a reserved Windows device name and cannot be "
            "used as a folder name or filename prefix."
        )

    if len(stripped) > _MAX_NAME_LEN:
        raise ValueError(
            f"Label {stripped!r} is {len(stripped)} characters long; folder and "
            f"file names must be at most {_MAX_NAME_LEN} characters."
        )

    return stripped


def validate_class_queries(class_queries: dict) -> dict:
    """
    Validate the whole {label: [queries]} mapping before any network calls.

    Failing early matters: a scrape can run for many minutes, and discovering a
    bad folder name after the fact means doing it all again.
    """
    if not isinstance(class_queries, dict):
        raise TypeError(
            "class_queries must be a dict mapping a class label to a list of "
            f"search queries, got {type(class_queries).__name__!r}"
        )
    if not class_queries:
        raise ValueError("class_queries is empty.")

    cleaned = {}
    for label, queries in class_queries.items():
        clean_label = validate_label_as_folder_name(label)

        if isinstance(queries, str):
            raise TypeError(
                f"Queries for class {clean_label!r} must be a list of strings, not a "
                f"single string. Did you mean [{queries!r}]?"
            )
        if not queries:
            raise ValueError(f"Class {clean_label!r} has no search queries.")

        clean_queries = []
        for i, q in enumerate(queries):
            if not isinstance(q, str) or not q.strip():
                raise ValueError(
                    f"Query {i} for class {clean_label!r} is empty or not a string: {q!r}"
                )
            clean_queries.append(q.strip())

        if len(set(clean_queries)) != len(clean_queries):
            raise ValueError(f"Class {clean_label!r} has duplicate queries.")

        cleaned[clean_label] = clean_queries

    return cleaned


# ================================================================
# 2. Image quality filters
# ================================================================

def validate_aspect_ratio_range(aspect_ratio_range):
    if not isinstance(aspect_ratio_range, (tuple, list)) or len(aspect_ratio_range) != 2:
        raise ValueError(
            "aspect_ratio_range must be a tuple/list of length 2, e.g. (0.5, 2.0)"
        )
    min_ratio, max_ratio = aspect_ratio_range
    if min_ratio <= 0 or max_ratio <= 0:
        raise ValueError("aspect_ratio_range values must be positive")
    if min_ratio > max_ratio:
        raise ValueError("aspect_ratio_range must satisfy min <= max")
    return float(min_ratio), float(max_ratio)


def validate_resolution_threshold(min_resolution):
    if not isinstance(min_resolution, (tuple, list)) or len(min_resolution) != 2:
        raise ValueError(
            "min_resolution must be a tuple/list of length 2, e.g. (200, 200)"
        )
    min_w, min_h = min_resolution
    if min_w <= 0 or min_h <= 0:
        raise ValueError("min_resolution values must be positive")
    return int(min_w), int(min_h)


def passes_image_filters(
    img: Image.Image,
    aspect_ratio_range=(0.5, 2.0),
    min_resolution=(200, 200),
) -> tuple[bool, list[str]]:
    """
    Return (passed, failed_reasons) for a single image. `failed_reasons` is an
    empty list when the image passes everything.

    These filters are measured on the image actually downloaded, not on the
    dimensions the search engine reported — those are frequently wrong.
    """
    min_ratio, max_ratio = validate_aspect_ratio_range(aspect_ratio_range)
    min_w, min_h = validate_resolution_threshold(min_resolution)

    w, h = img.size
    ratio = w / max(h, 1)
    reasons = []

    if w < min_w or h < min_h:
        reasons.append("resolution")
    if not (min_ratio <= ratio <= max_ratio):
        reasons.append("aspect_ratio")

    return len(reasons) == 0, reasons


# ================================================================
# 3. HTTP download with retry
# ================================================================
# Image URLs point at arbitrary websites, not at DuckDuckGo. Many will be slow,
# dead, rate-limited, or serving something that is not an image at all. Retrying
# with exponential backoff is what keeps a long scrape from dying on the first
# flaky host.

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]


def get_default_headers() -> dict:
    return {"User-Agent": random.choice(USER_AGENTS)}


def safe_get(url, headers=None, timeout=10, retries=3, backoff_base=2.0):
    """
    GET a URL, retrying on rate limits, server errors, and network failures.

    Returns the response on success. Raises the last exception if every attempt
    failed, so the caller can log the URL and move on.
    """
    if headers is None:
        headers = get_default_headers()

    last_exception = None

    for i in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)

            if resp.status_code == 200:
                return resp

            # 429 = rate limited, 5xx = server having a bad day. Both are worth
            # waiting out. Anything else (404, 403, ...) is not going to improve.
            if resp.status_code == 429 or resp.status_code in {500, 502, 503, 504}:
                sleep = (backoff_base ** i) + random.uniform(0.5, 2.0)
                time.sleep(sleep)
                continue

            resp.raise_for_status()

        except (Timeout, ConnectionError) as e:
            last_exception = e
            time.sleep((backoff_base ** i) + random.uniform(0.5, 2.0))

        except RequestException as e:
            last_exception = e
            time.sleep((backoff_base ** i) + random.uniform(0.5, 2.0))

    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"Gave up on {url} after {retries} attempts")


def download_image(url: str, headers=None, timeout=10) -> Image.Image | None:
    """
    Download a single image URL and return it as an RGB PIL image.

    Returns None (rather than raising) when the URL does not serve an image or
    the bytes will not open — both are normal and common when scraping.
    """
    resp = safe_get(url, headers=headers, timeout=timeout)

    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return None

    try:
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        # Truncated downloads, formats PIL cannot read, HTML error pages served
        # with an image content-type — all land here.
        return None


# ================================================================
# 4. Searching DuckDuckGo
# ================================================================

def search_images(
    query: str,
    max_results: int = 100,
    safesearch: str = "moderate",
    region: str = "us-en",
    retries: int = 4,
    backoff_base: float = 3.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run one DuckDuckGo image search and return the raw result records.

    Each record is a dict with keys: image (full-size URL), thumbnail, url
    (the page the image was found on), title, width, height, source.

    DuckDuckGo rate-limits aggressively, and when it does it reports
    "No results found" rather than an explicit rate-limit error — which looks
    exactly like a query that genuinely matched nothing. Because a real query
    almost never returns nothing, this function retries with backoff before
    believing it. Without the retry a whole class can silently end up empty.

    Parameters
    ----------
    query : str
        The search term.
    max_results : int, default 100
        How many results to ask for. DuckDuckGo often returns fewer.
    safesearch : {"on", "moderate", "off"}, default "moderate"
        Content filter. Leave this on unless you have a reason not to;
        "off" will pull in material you do not want in a course dataset.
    region : str, default "us-en"
        Region/language code. Changing this changes what you get back —
        worth noting in your writeup if you use something other than the default.
    retries : int, default 4
        How many times to retry an empty or failed search before giving up.
    backoff_base : float, default 3.0
        Base of the exponential backoff between retries, in seconds.
    """
    for attempt in range(retries):
        try:
            results = list(DDGS().images(
                query=query,
                max_results=max_results,
                safesearch=safesearch,
                region=region,
            ))
            if results:
                return results
            reason = "no results"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"

        if attempt < retries - 1:
            sleep = (backoff_base ** attempt) + random.uniform(1.0, 3.0)
            if verbose:
                print(f"    {reason} — retrying in {sleep:.1f}s "
                      f"({attempt + 1}/{retries - 1})")
            time.sleep(sleep)

    if verbose:
        print(f"    search returned nothing for {query!r} after {retries} attempts")
    return []


# ================================================================
# 5. Scraping one class
# ================================================================

def scrape_class(
    label: str,
    queries: list[str],
    save_dir: Path,
    images_per_class: int = 250,
    results_per_query: int = 200,
    aspect_ratio_range: tuple = (0.5, 2.0),
    min_resolution: tuple = (200, 200),
    per_image_timeout: int = 10,
    safesearch: str = "moderate",
    region: str = "us-en",
    query_delay: tuple = (2.0, 5.0),
    verbose: bool = True,
) -> list[dict]:
    """
    Scrape every query for one class into save_dir/<label>/ and return one
    record per saved image.

    Queries are searched in order and stop early once images_per_class images
    have been saved, so put your best query first.
    """
    class_dir = save_dir / label
    class_dir.mkdir(parents=True, exist_ok=True)

    records = []
    seen_urls = set()   # dedupe across queries — overlapping queries return overlapping images
    headers = get_default_headers()

    for qi, query in enumerate(queries):
        if len(records) >= images_per_class:
            break

        if verbose:
            print(f"  query {qi + 1}/{len(queries)}: {query!r}")

        results = search_images(
            query,
            max_results=results_per_query,
            safesearch=safesearch,
            region=region,
        )

        if verbose:
            print(f"    {len(results)} results returned")

        for result in results:
            if len(records) >= images_per_class:
                break

            image_url = result.get("image")
            if not image_url or image_url in seen_urls:
                continue
            seen_urls.add(image_url)

            try:
                img = download_image(image_url, headers=headers, timeout=per_image_timeout)
            except Exception:
                continue    # dead host, timeout, refused connection — expected

            if img is None:
                continue

            passed, _reasons = passes_image_filters(
                img,
                aspect_ratio_range=aspect_ratio_range,
                min_resolution=min_resolution,
            )
            if not passed:
                continue

            # Save the image itself. Do NOT round-trip through matplotlib —
            # that re-renders the picture into a figure and changes both the
            # resolution and the framing of what you are training on.
            filename = f"{label}_{len(records):04d}.jpg"
            img.save(class_dir / filename, format="JPEG", quality=92)

            records.append({
                "filename": filename,
                "label": label,
                "query": query,
                "image_url": image_url,
                "source_url": result.get("url", ""),
                "title": result.get("title", ""),
                "width": img.size[0],
                "height": img.size[1],
            })

            if verbose and len(records) % 25 == 0:
                print(f"    saved {len(records)}/{images_per_class}")

        # Pause between queries so DuckDuckGo does not start refusing us.
        if qi < len(queries) - 1 and len(records) < images_per_class:
            time.sleep(random.uniform(*query_delay))

    if verbose:
        print(f"  -> {len(records)} images saved to {class_dir}")

    return records


# ================================================================
# 6. Building the whole dataset
# ================================================================

def build_image_dataset(
    class_queries: dict,
    save_dir: str | Path,
    images_per_class: int = 250,
    results_per_query: int = 200,
    aspect_ratio_range: tuple = (0.5, 2.0),
    min_resolution: tuple = (200, 200),
    per_image_timeout: int = 10,
    safesearch: str = "moderate",
    region: str = "us-en",
    query_delay: tuple = (2.0, 5.0),
    continue_on_error: bool = True,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Download a labeled image dataset from DuckDuckGo Images.

    One folder per class is created inside save_dir. Every class may have
    several search queries, and results from all of a class's queries land in
    the same folder — using several queries per class is how you get a class
    that looks like the whole category instead of one narrow slice of it.

    Parameters
    ----------
    class_queries : dict[str, list[str]]
        Maps each class label to its list of search queries. The label becomes
        the folder name and the filename prefix, so it must be a legal filename.
    save_dir : str or Path
        Root output directory. Created if it does not exist.
    images_per_class : int, default 250
        Stop saving once a class reaches this many images. Scrape well beyond
        the number you need to end up with — you will lose a meaningful fraction
        to junk and duplicates during cleaning.
    results_per_query : int, default 200
        How many search results to request per query. Not how many you keep:
        many will fail to download or fail the filters.
    aspect_ratio_range : tuple of (float, float), default (0.5, 2.0)
        Inclusive (min, max) bounds on width/height. Rejects extreme panoramas
        and very tall images, which are usually banners or infographics.
    min_resolution : tuple of (int, int), default (200, 200)
        Minimum acceptable (width, height) in pixels, measured on the downloaded
        image. Anything smaller is a thumbnail and will look terrible resized up.
    per_image_timeout : int, default 10
        Seconds to wait on a single image download before giving up on it.
    safesearch : {"on", "moderate", "off"}, default "moderate"
        Content filter.
    region : str, default "us-en"
        Region/language code for the search.
    query_delay : tuple of (float, float), default (2.0, 5.0)
        Random sleep range in seconds between queries, to avoid rate limiting.
    continue_on_error : bool, default True
        If True, a class that fails is logged and the run continues. If False,
        the first failure raises.
    verbose : bool, default True
        Print progress as the scrape runs.

    Returns
    -------
    records : list of dict
        One entry per saved image, with keys: filename, label, query,
        image_url, source_url, title, width, height. This is what you build
        your manifest CSV from.
    summary : dict
        Per-class counts and status, plus "total_saved" and "save_dir".
    """
    # Validate everything before touching the network.
    cleaned = validate_class_queries(class_queries)
    validate_aspect_ratio_range(aspect_ratio_range)
    validate_resolution_threshold(min_resolution)

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    summary = {"classes": {}, "total_saved": 0, "save_dir": str(save_dir)}

    for i, (label, queries) in enumerate(cleaned.items()):
        if verbose:
            print(f"\n[{i + 1}/{len(cleaned)}] class {label!r} — {len(queries)} queries")

        status, error_msg, records = "ok", None, []

        try:
            records = scrape_class(
                label=label,
                queries=queries,
                save_dir=save_dir,
                images_per_class=images_per_class,
                results_per_query=results_per_query,
                aspect_ratio_range=aspect_ratio_range,
                min_resolution=min_resolution,
                per_image_timeout=per_image_timeout,
                safesearch=safesearch,
                region=region,
                query_delay=query_delay,
                verbose=verbose,
            )
        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  FAILED: class {label!r} | {error_msg}")
            if not continue_on_error:
                raise

        # A class that finishes with nothing is almost always rate limiting, not
        # a genuinely unsearchable category. Flag it rather than reporting "ok",
        # so an empty class cannot slide past unnoticed.
        if status == "ok" and len(records) == 0:
            status = "empty"
            error_msg = "no images saved — likely rate limited; wait and re-run this class"

        all_records.extend(records)
        summary["classes"][label] = {
            "saved": len(records),
            "queries": len(queries),
            "status": status,
            "error": error_msg,
        }
        summary["total_saved"] += len(records)

    if verbose:
        print(f"\nDone. {summary['total_saved']} images saved to {save_dir}")
        for label, info in summary["classes"].items():
            flag = "" if info["status"] == "ok" else f"  [{info['status']}]"
            print(f"  {label:<24} {info['saved']:>5}{flag}")

    return all_records, summary
