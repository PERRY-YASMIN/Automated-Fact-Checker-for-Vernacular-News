import React from "react";

interface ClaimCardProps {
  claim: string;
  verdict?: string;
  confidence?: number;
  sources?: string[];
  onApprove?: () => void;
  onReject?: () => void;
}

const ClaimCard: React.FC<ClaimCardProps> = ({
  claim, verdict, confidence, sources, onApprove, onReject
}) => {
  return (
    <div className="claim-card">
      <p>{claim}</p>
      {verdict && (
        <div className="verdict">
          <span><b>Verdict:</b> {verdict}</span>
          <span>Confidence: {((confidence ?? 0) * 100).toFixed(2)}%</span>
        </div>
      )}
      {sources && sources.length > 0 && (
        <div style={{ marginTop: "8px" }}>
          <b>Sources:</b>
          <ul style={{ marginTop: "4px", marginBottom: 0 }}>
            {sources.map((source, index) => (
              <li key={index}>{source}</li>
            ))}
          </ul>
        </div>
      )}
      {(onApprove || onReject) && (
        <div className="buttons">
          {onApprove && <button className="approve" onClick={onApprove}>Approve</button>}
          {onReject && <button className="reject" onClick={onReject}>Reject</button>}
        </div>
      )}
    </div>
  );
};

export default ClaimCard;
