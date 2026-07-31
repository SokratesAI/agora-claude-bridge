import express from "express";
import pino from "pino";

const logger = pino();
const app = express();
const port = Number(process.env.PORT ?? 8080);

app.get("/healthz", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

app.listen(port, () => {
  logger.info({ port }, "service listening");
});
