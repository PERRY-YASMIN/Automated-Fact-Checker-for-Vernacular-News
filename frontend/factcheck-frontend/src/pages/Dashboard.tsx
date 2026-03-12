import React from "react";
import Header from "../components/Header";
import SearchBar from "../components/SearchBar";
import ClaimCard from "../components/ClaimCard";
import "../index.css";

interface VerificationResult {
  claim: string;
  verdict: string;
  confidence: number;
  sources: string[];
}

const Dashboard: React.FC = () => {
  const [claims, setClaims] = React.useState<VerificationResult[]>([]);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const handleSearch = (result: VerificationResult) => {
    setClaims((prev) => [result, ...prev]);
  };

  return (
    <div className="dashboard-bg">
      <Header />
      <SearchBar
        onSearch={handleSearch}
        onLoadingChange={setIsLoading}
        onError={setError}
      />
      {isLoading && (
        <p style={{ textAlign: "center", marginTop: "10px" }}>Verifying claim...</p>
      )}
      {error && (
        <p style={{ textAlign: "center", marginTop: "10px", color: "#d93025" }}>{error}</p>
      )}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        {claims.map((claim, idx) => (
          <ClaimCard
            key={idx}
            claim={claim.claim}
            verdict={claim.verdict}
            confidence={claim.confidence}
            sources={claim.sources}
            onApprove={() => alert("Approved!")}
            onReject={() => alert("Rejected!")}
          />
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
