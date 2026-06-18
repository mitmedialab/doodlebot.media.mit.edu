"""Static HTML pages: phone drawing UI, curator gallery, big-screen display.

The HTML files are not shipped with the server. On the first request for a page
we fetch it from STATIC_HOST, cache it under STATIC_DIR, and serve the local copy
on every subsequent request. POST /static/bust clears the cache so a redeployed
front-end is picked up on the next page load.
"""

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from .common import require_admin
from .config import STATIC_DIR, STATIC_HOST

router = APIRouter()


async def _serve(filename: str) -> FileResponse:
    """Serve a cached page, fetching it from STATIC_HOST on first use."""
    path = STATIC_DIR / filename
    if not path.exists():
        url = f"{STATIC_HOST.rstrip('/')}/{filename}"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Could not fetch {filename} from {url}: {e}")
        # Write atomically so a concurrent request never serves a partial file.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
    return FileResponse(path)


@router.get("/")
async def draw_page() -> FileResponse:
    return await _serve("index.html")


@router.get("/gallery")
async def gallery_page() -> FileResponse:
    return await _serve("gallery.html")


@router.get("/display")
async def display_page() -> FileResponse:
    return await _serve("display.html")


@router.get("/vectorization")
async def vectorization_page() -> FileResponse:
    return await _serve("vectorization.html")


@router.get("/robots")
async def robots_page() -> FileResponse:
    return await _serve("robots.html")


@router.get("/robot.png")
async def robot_image() -> FileResponse:
    return await _serve("robot.png")


@router.api_route("/static/bust", methods=["GET", "POST"])
async def bust_static(req: Request) -> RedirectResponse:
    """Clear the cached static files so they're re-fetched on the next request.

    Exposed over GET so it works by just navigating to
    /static/bust?token=<ADMIN_TOKEN> in the browser; afterwards we redirect to
    the page given by ?next= (default "/") which re-fetches the fresh copy.
    """
    require_admin(req)
    for path in STATIC_DIR.iterdir():
        if path.is_file():
            path.unlink()
    next_url = req.query_params.get("next", "/")
    return RedirectResponse(next_url, status_code=303)
