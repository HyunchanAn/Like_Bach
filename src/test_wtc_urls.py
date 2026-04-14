import requests

def test_wtc_url():
    # Test Book 1
    url = "https://kern.humdrum.org/cgi-bin/ksdata?l=bach/wtc1&format=kern&file=wtc1f01.krn"
    print(f"Testing {url}...")
    resp = requests.get(url)
    print(f"Status: {resp.status_code}, Length: {len(resp.text)}")
    print(resp.text[:100])
    
    # Test Book 2
    url2 = "https://kern.humdrum.org/cgi-bin/ksdata?l=bach/wtc2&format=kern&file=wtc2f01.krn"
    print(f"Testing {url2}...")
    resp2 = requests.get(url2)
    print(f"Status: {resp2.status_code}, Length: {len(resp2.text)}")
    print(resp2.text[:100])

if __name__ == "__main__":
    test_wtc_url()
