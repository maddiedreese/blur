# Privacy

Blur performs classification inside the Chrome extension runtime. It has no developer-controlled server, analytics, telemetry, remote inference, or external API integration.

## Data flow

The content script sends the selected image URL to the extension service worker. An extension-owned offscreen document retrieves the bytes of that image from the same URL the page displays, inspects bounded metadata, decodes the pixels, and runs the bundled model locally. Only the resulting score and short evidence labels return to the page overlay.

Blur does not send image bytes, page URLs, image URLs, scores, metadata, or browsing activity to its developer or any inference provider. Model and runtime assets resolve from the extension package. Images and their metadata are discarded after scoring.

## Permissions

- `offscreen`: owns the browser-local image decoder and inference session.
- `storage`: stores the fixed settings and per-origin enable/disable choice.
- `http://*/*` and `https://*/*`: discovers images on ordinary webpages and retrieves their bytes for local analysis across origins.

The extension does not request `tabs`, `webRequest`, `cookies`, `history`, `downloads`, `nativeMessaging`, or `unlimitedStorage`.
