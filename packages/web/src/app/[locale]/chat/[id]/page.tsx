import { ChatPage } from '@/components/chat/ChatPage';

export default async function ChatConversationRoute({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ChatPage conversationId={id} />;
}
