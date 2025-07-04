import * as dotenv from 'dotenv';

dotenv.config();

import { genkit } from "genkit/beta";
import { defineCodeFormat } from "./code-format.js";
import { vertexAI } from "@genkit-ai/vertexai";

if (!process.env.CLIENT_EMAIL || !process.env.PRIVATE_KEY || !process.env.PROJECT_ID || !process.env.LOCATION) {
  throw new Error("Missing environment variables");
}

export const ai = genkit({
  plugins: [
    vertexAI({
      googleAuth: {
        credentials: {
          client_email: process.env.CLIENT_EMAIL,
          private_key: process.env.PRIVATE_KEY,
        }
      },
      projectId: process.env.PROJECT_ID,
      location: process.env.LOCATION,
    })
  ],
  model: vertexAI.model("gemini-2.0-flash"),
});

defineCodeFormat(ai);

export { z } from "genkit/beta";
