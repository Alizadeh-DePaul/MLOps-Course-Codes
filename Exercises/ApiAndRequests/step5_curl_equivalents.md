# Step 5 - curl equivalents

Every `requests` call in steps 1-4 has a one-line `curl` equivalent. Knowing
both is useful: Python is what your application speaks, `curl` is what you
type when you're SSH'd into a box at 2 AM and a deploy is broken.

> **Windows PowerShell users:** PowerShell aliases `curl` to
> `Invoke-WebRequest`, which is a different tool with different flags. Type
> `curl.exe` (with the extension) to force the real curl binary. On Windows
> 10 (1803+) and Windows 11, `curl.exe` is pre-installed - no separate
> download needed. Verify with `curl.exe --version`.

## GET — fetch a URL

```python
# Python
response = requests.get("https://api.github.com")
print(response.status_code)
```

```bash
# curl - full body
curl -X GET "https://api.github.com"

# curl - headers only (status + headers, no body)
curl -X GET -I "https://api.github.com"
```

The `-I` flag (capital i) tells curl to only fetch HTTP headers — useful for
a quick "is this URL alive?" check without downloading the response body.

## GET with query parameters

```python
# Python
response = requests.get(
    "https://api.github.com/search/repositories",
    params={"q": "requests+language:python"},
)
```

```bash
# curl - you have to URL-encode the query yourself
curl -X GET "https://api.github.com/search/repositories?q=requests+language:python"
```

`requests` handles URL encoding for you when you pass `params=`. With curl,
you have to know the encoding rules.

## GET binary - save to a file

```python
# Python
response = requests.get(IMG_URL)
with open("img.png", "wb") as f:
    f.write(response.content)
```

```bash
# curl - write to a file directly
curl -o img.png "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png"
```

The `-o` flag tells curl to save the response body to the named file.

## POST form-encoded

```python
# Python
response = requests.post(
    "https://httpbin.org/post",
    data={"username": "depaul", "password": "se489"},
)
```

```bash
# curl - --data sends form-encoded (just like requests' data= kwarg)
curl -X POST "https://httpbin.org/post" \
     --data "username=depaul&password=se489"
```

## POST JSON

```python
# Python
response = requests.post(
    "https://httpbin.org/post",
    json={"username": "depaul", "password": "se489"},
)
```

```bash
# curl - you set Content-Type AND serialize the body yourself
curl -X POST "https://httpbin.org/post" \
     -H "Content-Type: application/json" \
     -d '{"username":"depaul","password":"se489"}'
```

This is the form vs JSON gotcha (step 4) again. `requests.post(..., json=...)`
does two things for you: sets the header and JSON-encodes the body. With curl
you have to do both by hand.

## A useful curl flag you'll want later

`-v` (verbose) shows the full request line, the request headers, the response
headers, AND the body. When something is broken and the error message is
useless, `-v` is the first thing to add:

```bash
curl -v -X POST "https://httpbin.org/post" -d "hello=world"
```

The lines starting with `>` are the request, lines starting with `<` are the
response. This is the closest thing to a packet capture you can do from your
terminal.
