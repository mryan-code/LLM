const llmHost = process.env.LLM_HOST || '127.0.0.1';
const llmPort = process.env.LLM_PORT || '8765';
const pm2AppName = process.env.PM2_APP || 'sz-llm';
const seedFlag = (process.env.STORM_ZERO_SEED_FOUNDATION || '').toLowerCase();

const args = ['-m', 'storm_zero_llm', '--host', llmHost, '--port', String(llmPort)];
if (['1', 'true', 'yes', 'on'].includes(seedFlag)) {
  args.push('--seed-foundation');
}

module.exports = {
  apps: [
    {
      name: pm2AppName,
      script: './.venv/bin/python',
      args,
      cwd: __dirname,
      env: {
        PM2_APP: pm2AppName,
        LLM_HOST: llmHost,
        LLM_PORT: String(llmPort),
        STORM_ZERO_SEED_FOUNDATION: process.env.STORM_ZERO_SEED_FOUNDATION || 'false'
      }
    }
  ]
};
