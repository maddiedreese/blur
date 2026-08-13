import { describe, expect, it } from 'vitest';
import { classifyManifestStore } from '../src/inference/provenance';

const trained = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia';
const composited = 'http://cv.iptc.org/newscodes/digitalsourcetype/compositedWithTrainedAlgorithmicMedia';

function store(validation_state: 'Invalid' | 'Valid' | 'Trusted', digitalSourceType: string) {
  return {
    validation_state,
    active_manifest: 'active',
    manifests: {
      active: { assertions: [{ label: 'c2pa.actions.v2', data: { actions: [{ action: 'c2pa.created', digitalSourceType }] } }] },
    },
  };
}

describe('verified C2PA interpretation', () => {
  it('uses validated trained-algorithmic creation as decisive evidence', () => {
    const result = classifyManifestStore(store('Valid', trained));
    expect(result.aiCreated).toBe(true);
    expect(result.scoreFloor).toBe(0.995);
    expect(result.trustedSigner).toBe(false);
  });

  it('distinguishes local cryptographic validity from signer trust', () => {
    expect(classifyManifestStore(store('Trusted', trained)).trustedSigner).toBe(true);
  });

  it('assigns the documented floor to validated AI-composited media', () => {
    const result = classifyManifestStore(store('Valid', composited));
    expect(result.aiCreated).toBe(false);
    expect(result.aiModified).toBe(true);
    expect(result.scoreFloor).toBe(0.9);
  });

  it('never trusts an assertion from an invalid manifest', () => {
    const result = classifyManifestStore(store('Invalid', trained));
    expect(result.aiCreated).toBe(false);
    expect(result.scoreFloor).toBe(0);
    expect(result.warning).toMatch(/did not validate/);
  });
});
