This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to work on the UI. Changes in `src/app` hot‑reload automatically.

## Static Export → FastAPI

The backend serves a fully static export from `backend/static_frontend`. Build and sync via:

```bash
# from frontend/
npm run deploy:static
```

This runs `next build` (with `output: "export"`) and copies the generated `frontend/out` directory into `backend/static_frontend`. If you prefer manual steps:

```bash
npm run build:static   # produces frontend/out
npm run sync:backend   # mirrors frontend/out -> backend/static_frontend
```

After syncing, start the FastAPI app (e.g., `uvicorn backend.fastapi_app:web_app --reload`) and hit `http://localhost:8000/` to confirm the exported UI loads alongside the `/voice_ws` and `/editor-state` APIs.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
