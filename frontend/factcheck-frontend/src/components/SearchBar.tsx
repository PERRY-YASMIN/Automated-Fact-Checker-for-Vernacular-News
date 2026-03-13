import React from "react";

interface VerifyResponse {
  claim: string;
  verdict: string;
  confidence: number;
  sources: Array<string | { claim?: string; id?: string }>;
}

interface NormalizedVerifyResponse {
  claim: string;
  verdict: string;
  confidence: number;
  sources: string[];
}

interface SearchBarProps {
  onSearch?: (result: NormalizedVerifyResponse) => void;
  onLoadingChange?: (isLoading: boolean) => void;
  onError?: (message: string | null) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch, onLoadingChange, onError }) => {
  const [query, setQuery] = React.useState("");
  const REQUEST_TIMEOUT_MS = 90000;

  const runVerification = async () => {
    const text = query.trim();
    if (!text) {
      onError?.("Please enter a claim to verify.");
      return;
    }

    onError?.(null);
    onLoadingChange?.(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch("/verify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data: VerifyResponse = await response.json();
      const normalizedSources = (data.sources ?? []).map((source) => {
        if (typeof source === "string") {
          return source;
        }
        return source.claim ?? source.id ?? "Unknown source";
      });

      onSearch?.({
        claim: data.claim,
        verdict: data.verdict,
        confidence: data.confidence,
        sources: normalizedSources,
      });
      setQuery("");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        onError?.("Verification timed out. First request can be slower while models warm up. Please try again.");
      } else {
        onError?.("Could not verify the claim. Please try again.");
      }
    } finally {
      clearTimeout(timeout);
      onLoadingChange?.(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      void runVerification();
    }
  };

  return (
    <input
      type="text"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      onKeyPress={handleKeyPress}
      placeholder="Search claims or posts..."
      className="search-bar"
    />
  );
};

export default SearchBar;
