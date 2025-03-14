"use client"; // ✅ Move this to the top

import { ApolloProvider } from "@apollo/client";
import client from "../lib/apolloClient";
import "../globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <title>LLM‑Powered Fraud Detection</title>
      </head>
      <body className="bg-gray-50 text-gray-900">
        <ApolloProvider client={client}>
          {children}
        </ApolloProvider>
      </body>
    </html>
  );
}
