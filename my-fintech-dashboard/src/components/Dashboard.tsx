"use client";

import React from "react";
import { useQuery, gql } from "@apollo/client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface Transaction {
  transaction_id: string;
  fraud_score: number;
  label: string;
}

const GET_TRANSACTIONS = gql`
  query GetTransactions {
    transactions {
      transaction_id
      fraud_score
      label
    }
  }
`;

const Dashboard: React.FC = () => {
  const { data, loading, error } = useQuery<{ transactions: Transaction[] }>(
    GET_TRANSACTIONS,
    { pollInterval: 5000 }
  );

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500"></div>
      </div>
    );
  }

  const chartData =
    data?.transactions.map((tx, index) => ({
      name: `Tx ${index + 1}`,
      fraud_score: tx.fraud_score,
    })) || [];

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      {/* Header */}
      <h1 className="text-3xl font-bold text-center text-blue-600 mb-6">
        LLM‑POWERED FRAUD DETECTION & ANOMALY ANALYSIS
      </h1>

      {/* Fraud Score Trend Chart */}
      <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-3 text-gray-800">
          Fraud Score Trend
        </h2>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line type="monotone" dataKey="fraud_score" stroke="#ff6347" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500 text-center">No data available</p>
        )}
      </div>

      {/* Alert Logs */}
      <div className="bg-white shadow-lg rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-3 text-gray-800">
          Alert Logs
        </h2>
        {data?.transactions.length ? (
          <ul className="divide-y divide-gray-200">
            {data.transactions.map((tx) => (
              <li
                key={tx.transaction_id}
                className="py-3 flex justify-between text-gray-700"
              >
                <span className="font-medium text-gray-900">
                  {tx.transaction_id}
                </span>
                <span className="text-sm">
                  Fraud Score:{" "}
                  <span
                    className={`px-2 py-1 rounded ${
                      tx.fraud_score > 0.7 ? "bg-red-500 text-white" : "bg-green-500 text-white"
                    }`}
                  >
                    {tx.fraud_score.toFixed(2)}
                  </span>
                </span>
                <span className="text-sm text-gray-600">{tx.label}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500 text-center">No transactions found</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
