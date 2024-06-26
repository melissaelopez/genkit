import { embed, EmbedderArgument, embedMany } from '@genkit-ai/ai/embedder';
import {
  CommonRetrieverOptionsSchema,
  defineIndexer,
  defineRetriever,
} from '@genkit-ai/ai/retriever';
import * as aiplatform from '@google-cloud/aiplatform';
import { IndexServiceClient } from '@google-cloud/aiplatform';

import { GoogleAuth } from 'google-auth-library';
import z from 'zod';
import { PluginOptions } from '.';

type UpsertRequest =
  aiplatform.protos.google.cloud.aiplatform.v1.IUpsertDatapointsRequest;

type IIndexDatapoint =
  aiplatform.protos.google.cloud.aiplatform.v1.IIndexDatapoint;

class Datapoint extends aiplatform.protos.google.cloud.aiplatform.v1
  .IndexDatapoint {
  constructor(properties: IIndexDatapoint) {
    super(properties);
  }
}

interface VVSOptions<EmbedderCustomOptions extends z.ZodTypeAny> {
  pluginOptions: PluginOptions;
  authClient: GoogleAuth;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}

const VVSOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000),
});

export function configureVVSIndexer<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: VVSOptions<EmbedderCustomOptions>
) {
  const { documentIndexer, documentIdField } =
    params.pluginOptions.vectorSearchOptions!;
  const { embedder, embedderOptions } = params;
  const indexServiceClient = new IndexServiceClient({
    auth: params.authClient,
  });

  const indexId = params.pluginOptions.vectorSearchOptions!.index;

  return defineIndexer(
    {
      name: `vertexai/${indexId}`,
      configSchema: z.any(),
    },
    async (docs, options) => {
      const ids = docs.map((doc) => doc.metadata![documentIdField!]);

      const embeddings = await embedMany({
        embedder,
        content: docs,
      });

      const datapoints = embeddings.map(
        ({ embedding }, i) =>
          new Datapoint({
            datapointId: ids[i],
            featureVector: embedding,
          })
      );

      const upsertRequest: UpsertRequest = {
        datapoints,
        index: indexId,
      };
      try {
        await indexServiceClient.upsertDatapoints(upsertRequest);
        await documentIndexer(docs);
      } catch (error) {
        console.error(error);
      }
    }
  );
}

export function configureVVSRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(params: VVSOptions<EmbedderCustomOptions>) {
  const { documentRetriever, documentIdField } =
    params.pluginOptions.vectorSearchOptions!;
  const indexId = params.pluginOptions.vectorSearchOptions!.index;
  return defineRetriever(
    {
      name: `vertexai/${indexId}`,
      configSchema: VVSOptionsSchema,
    },
    async (content, options) => {
      const queryEmbeddings = await embed({
        embedder: params.embedder,
        content,
        options: params.embedderOptions,
      });

      const accessToken = await params.authClient.getAccessToken();
      const projectId = params.pluginOptions.projectId!;
      const location = params.pluginOptions.location!;
      const publicEndpointDomainName =
        params.pluginOptions.vectorSearchOptions!.publicEndpoint;

      try {
        const queryResponse = await queryPublicEndpoint({
          featureVector: queryEmbeddings,
          neighborCount: options.k,
          accessToken: accessToken!,
          projectId,
          location,
          publicEndpointDomainName,
          deployedIndexId: indexId,
        });

        const docIds = queryResponse.queries[0].neighbors.map((n) => {
          return n.datapoint.datapointId;
        });

        const documentResponse = await documentRetriever(docIds);

        return {
          documents: documentResponse,
        };
      } catch (error) {
        console.error(error);
        return {
          documents: [],
        };
      }
    }
  );
}

interface QueryPublicEndpointParams {
  featureVector: number[];
  neighborCount: number;
  accessToken: string;
  projectId: string;
  location: string;
  deployedIndexId: string;
  publicEndpointDomainName: string;
}

async function queryPublicEndpoint(
  params: QueryPublicEndpointParams
): Promise<any> {
  const {
    featureVector,
    neighborCount,
    accessToken,
    deployedIndexId,
    publicEndpointDomainName,
  } = params;
  const url = publicEndpointDomainName;

  const requestBody = {
    deployed_index_id: deployedIndexId,
    queries: [
      {
        datapoint: {
          datapoint_id: '0',
          feature_vector: featureVector,
        },
        neighbor_count: neighborCount,
      },
    ],
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    throw new Error(`Error: ${response.statusText}`);
  }

  return await response.json();
}
