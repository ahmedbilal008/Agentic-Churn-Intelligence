// MCP-to-HTTP bridge: calls the MCP server for a specific tool
// Route: POST /api/mcp/[tool]
//
// Uses the official @modelcontextprotocol/sdk Client + SSEClientTransport
// to properly manage the SSE session lifecycle and avoid ClosedResourceError.

import { NextRequest, NextResponse } from "next/server";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

const MCP_BASE_URL = (process.env.NEXT_PUBLIC_MCP_URL || "http://localhost:8000").replace(/\/$/, "");

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ tool: string }> }
) {
  let client: Client | null = null;
  let transport: SSEClientTransport | null = null;

  try {
    const { tool } = await params;
    const body = await request.json();
    const args = body.args || {};

    // Create an MCP client with proper SSE transport
    transport = new SSEClientTransport(new URL(`${MCP_BASE_URL}/sse`));
    client = new Client({
      name: "churn-ui",
      version: "1.0.0",
    });

    // Connect (performs initialize + initialized handshake)
    await client.connect(transport);

    // List tools so the SDK registers them — suppresses
    // "Tool not listed, no validation will be performed" warnings
    await client.listTools();

    // Call the tool
    const result = await client.callTool({
      name: tool,
      arguments: args,
    });

    // Extract text content from MCP response
    let parsed: unknown = result;
    const content = result?.content;
    if (Array.isArray(content) && content.length > 0) {
      const text = (content[0] as { text?: string })?.text;
      if (text) {
        try {
          parsed = JSON.parse(text);
        } catch {
          parsed = text;
        }
      }
    }

    return NextResponse.json({ result: parsed, status: "success" });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("MCP bridge error:", message);
    return NextResponse.json(
      { error: message, status: "error" },
      { status: 500 }
    );
  } finally {
    // Properly close the transport to avoid ClosedResourceError on server
    try {
      await transport?.close();
    } catch {
      // ignore close errors
    }
  }
}
