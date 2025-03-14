import { ApolloClient, InMemoryCache } from "@apollo/client";

const client = new ApolloClient({
  uri: "http://localhost:5000/graphql", // Make sure this matches your backend URL
  cache: new InMemoryCache(),
});

export default client;
